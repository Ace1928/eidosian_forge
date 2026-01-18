import collections
import functools
import os
import sys
from absl import logging
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.capture import restore_captures
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import restore
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.polymorphic_function import saved_model_utils as function_saved_model_utils
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class Loader(object):
    """Helper class to load an object-based SavedModel."""

    def __init__(self, object_graph_proto, saved_model_proto, export_dir, ckpt_options, save_options, filters):
        meta_graph = saved_model_proto.meta_graphs[0]
        self._asset_file_def = meta_graph.asset_file_def
        self._operation_attributes = {node.name: node.attr for node in meta_graph.graph_def.node}
        self._proto = object_graph_proto
        self._export_dir = export_dir
        self._concrete_functions = function_deserialization.load_function_def_library(library=meta_graph.graph_def.library, saved_object_graph=self._proto, wrapper_function=_WrapperFunction)
        self._restored_concrete_functions = set()
        self._checkpoint_options = ckpt_options
        self._save_options = save_options
        self._concrete_function_aliases = meta_graph.meta_info_def.function_aliases
        self.function_aliases = {}
        if self._save_options.experimental_load_function_aliases:
            concrete_func_list_by_alias = collections.defaultdict(list)
            for concrete_func_name, alias in self._concrete_function_aliases.items():
                if concrete_func_name not in self._concrete_functions:
                    logging.warn('ConcreteFunction `%s` is listed in function alias but it is not found.', concrete_func_name)
                    continue
                concrete_function = self._concrete_functions[concrete_func_name]
                concrete_func_list_by_alias[alias].append(concrete_function)
            self.function_aliases = dict(concrete_func_list_by_alias)
        self._pretty_printer = checkpoint.ObjectGraphProtoPrettyPrinter(self._proto)
        self._node_filters = filters
        self._node_path_to_id = self._convert_node_paths_to_ints()
        self._loaded_nodes = {}
        if isinstance(filters, dict):
            for node_path, node in filters.items():
                if isinstance(node, tuple):
                    self._loaded_nodes[self._node_path_to_id[node_path]] = node
                else:
                    self._loaded_nodes[self._node_path_to_id[node_path]] = (node, setattr)
        self._filtered_nodes = self._retrieve_all_filtered_nodes()
        self._ordered_node_ids = self._generate_ordered_node_ids()
        self._load_all()
        if not save_options.experimental_skip_checkpoint:
            self._restore_checkpoint()
        for node in self._nodes:
            if isinstance(node, resource.CapturableResource):
                init_op = node._initialize()
                if not context.executing_eagerly():
                    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)

    def _convert_node_paths_to_ints(self):
        """Maps all string node paths in node_filters to the int node ids."""
        if self._node_filters is None:
            return None
        path_to_int = {}
        for node_id in self._node_filters:
            int_node_id = None
            if isinstance(node_id, str):
                node_path = node_id.split('.')
                if node_path[0] != 'root':
                    raise ValueError(f'When passing string identifiers to node_filters, the first name must be root. Received {node_path[0]}.')
                int_node_id = 0
                for n, name in enumerate(node_path[1:]):
                    int_node_id = self._find_node_child(int_node_id, name, '.'.join(node_path[:n + 2]))
                path_to_int[node_id] = int_node_id
            else:
                raise TypeError('Elements in node_filters must be strings.')
        return path_to_int

    def _retrieve_all_filtered_nodes(self):
        """Traverses through the object graph to get the IDs of all nodes to load.

    As a side-effect, if node_filters is a dictionary that contains already-
    created objects, then the children tracked by those objects will be
    added to node_filters.

    Returns:
      List of all nodes to load, or None if all nodes should be loaded.

    """
        if self._node_filters is None:
            return None
        all_filtered_nodes = set()
        nodes_to_visit = list(self._node_filters)
        while nodes_to_visit:
            node_path = nodes_to_visit.pop(0)
            node_id = self._node_path_to_id[node_path]
            if node_id in all_filtered_nodes:
                continue
            all_filtered_nodes.add(node_id)
            node, setter = self._loaded_nodes.get(node_id, (None, None))
            if node is not None:
                if not isinstance(node, base.Trackable):
                    raise TypeError(f"Error when processing dictionary values passed to nodes_to_load.Object at {node_path} is expected to be a checkpointable (i.e. 'trackable') TensorFlow object (e.g. tf.Variable, tf.Module or Keras layer).")
                node._maybe_initialize_trackable()
            for reference in self._proto.nodes[node_id].children:
                child_object, _ = self._loaded_nodes.get(reference.node_id, (None, None))
                if child_object is None and node is not None:
                    child_object = node._lookup_dependency(reference.local_name)
                    if isinstance(child_object, data_structures.TrackableDataStructure):
                        setter = lambda *args: None
                        self._loaded_nodes[reference.node_id] = (child_object, setter)
                child_path = '{}.{}'.format(node_path, reference.local_name)
                self._node_path_to_id[child_path] = reference.node_id
                nodes_to_visit.append(child_path)
        if 0 in all_filtered_nodes:
            return None
        return all_filtered_nodes

    def _find_node_child(self, node_id, child_name, path):
        for reference in self._proto.nodes[node_id].children:
            if reference.local_name == child_name:
                return reference.node_id
        raise ValueError(f'Unable to find node {path}.')

    def _load_all(self):
        """Loads all nodes and functions from the SavedModel and their edges."""
        self._load_nodes()
        self._load_edges()
        self._setup_remaining_functions()
        self._load_checkpoint_save_and_restore_functions()

    def _load_checkpoint_save_and_restore_functions(self):
        """Restores the checkpoint-related save/restore functions to all nodes."""
        temp_session = [None]
        for node_id, proto in self._iter_all_nodes():
            node = self.get(node_id)
            if proto.saveable_objects.keys() == {trackable_utils.SERIALIZE_TO_TENSORS_NAME}:
                assert len(proto.saveable_objects) == 1
                saveable_object_proto = next(iter(proto.saveable_objects.values()))
                save_fn_id = saveable_object_proto.save_function
                restore_fn_id = saveable_object_proto.restore_function
                node._serialize_to_tensors = self.get(save_fn_id)
                node._restore_from_tensors = self.get(restore_fn_id)
            else:
                saveable_fn_by_name = {}
                for name, saveable_object_proto in proto.saveable_objects.items():
                    save_fn_id = saveable_object_proto.save_function
                    restore_fn_id = saveable_object_proto.restore_function
                    saveable_fn_by_name[name] = (self.get(save_fn_id), self.get(restore_fn_id))
                node._self_saveable_object_factories = saveable_object_util.recreate_saveable_objects(saveable_fn_by_name, temp_session)

    def _load_edges(self):
        """Adds edges from objects to other objects and functions."""
        for node_id, object_proto in self._iter_all_nodes():
            self._add_object_graph_edges(object_proto, node_id)
        if self._filtered_nodes is not None and 0 not in self._filtered_nodes:
            root = self.get(0)
            for node_path in self._node_filters:
                loaded_node = self._nodes[self._node_path_to_id[node_path]]
                path = node_path.split('.')
                current_node = root
                for name in path[1:-1]:
                    if not hasattr(current_node, name):
                        setattr(current_node, name, self._recreate_base_user_object()[0])
                    current_node = getattr(current_node, name)
                if not hasattr(current_node, path[-1]):
                    setattr(current_node, path[-1], loaded_node)

    def _add_object_graph_edges(self, proto, node_id):
        """Adds edges from an object to its children."""
        obj = self._nodes[node_id]
        setter = self._node_setters[node_id]
        for reference in proto.children:
            setter(obj, reference.local_name, self._nodes[reference.node_id])
            if reference.local_name == '__call__' and (not callable(obj)):
                setattr(type(obj), '__call__', _call_attribute)

    def _setup_remaining_functions(self):
        concrete_function_names = sorted(self._proto.concrete_functions.keys())
        for name in concrete_function_names:
            if name in self._restored_concrete_functions:
                continue
            self._setup_function_captures(name, self._nodes)

    def _setup_function_captures(self, concrete_function_name, nodes):
        """Setup captures and variables in a restored function."""
        if concrete_function_name in self._restored_concrete_functions:
            return
        self._restored_concrete_functions.add(concrete_function_name)
        concrete_function = self._concrete_functions[concrete_function_name]
        proto = self._proto.concrete_functions[concrete_function_name]
        inputs = [nodes[node_id] for node_id in proto.bound_inputs]
        restore_captures.restore_captures(concrete_function, inputs)

    def _initialize_loaded_nodes(self):
        nodes = {}
        node_setters = {}
        for node_id, (node, setter) in self._loaded_nodes.items():
            nodes[node_id] = node
            node_setters[node_id] = setter
        return (nodes, node_setters)

    def _get_node_dependencies(self, proto):
        """Returns a dictionary of all dependencies of an object.

    Args:
      proto: A SavedObject proto.

    Returns:
      Dict mapping string dependency name *or* int node id to the node id.
      The int node id key is used for mapping function captures.
    """
        dependencies = {ref.local_name: ref.node_id for ref in proto.dependencies}
        kind = proto.WhichOneof('kind')
        if kind == 'function':
            concrete_functions = proto.function.concrete_functions
            for fn_name in concrete_functions:
                for bound_input in self._proto.concrete_functions[fn_name].bound_inputs:
                    dependencies[bound_input] = bound_input
        elif kind == 'bare_concrete_function':
            fn_name = proto.bare_concrete_function.concrete_function_name
            for bound_input in self._proto.concrete_functions[fn_name].bound_inputs:
                dependencies[bound_input] = bound_input
        elif kind == 'resource':
            for child in proto.children:
                if child.local_name == '_create_resource':
                    dependencies['_create_resource'] = child.node_id
        return dependencies

    def _generate_ordered_node_ids(self):
        """Orders the node ids so that dependencies appear first."""
        if self._filtered_nodes is None:
            unordered_ids = range(len(self._proto.nodes))
        else:
            unordered_ids = list(self._filtered_nodes)
        dependency_map = collections.defaultdict(list)
        for node_id in unordered_ids:
            deps = dependency_map[node_id]
            if self._loaded_nodes.get(node_id) is not None:
                continue
            proto = self._proto.nodes[node_id]
            for dep in set(self._get_node_dependencies(proto).values()):
                deps.append(dep)
                if self._filtered_nodes is not None and dep not in self._filtered_nodes:
                    raise ValueError(f'Unable to partially load SavedModel since the specified filter does not include all required objects for loading (e.g. variables used in functions or deserialization dependencies). Please include this path in the filter: {self._pretty_printer.node_names[dep]}')
            prev_slot = None
            for slot_variable_proto in proto.slot_variables:
                slot_variable_node_id = slot_variable_proto.slot_variable_node_id
                slot_deps = dependency_map[slot_variable_node_id]
                slot_deps.append(node_id)
                slot_deps.append(slot_variable_proto.original_variable_node_id)
                if prev_slot is not None:
                    slot_deps.append(prev_slot)
                prev_slot = slot_variable_node_id
        try:
            return list(trackable_utils.order_by_dependency(dependency_map))
        except trackable_utils.CyclicDependencyError:
            raise ValueError('Encountered a cycle in the deserialization dependenciesin the SavedModel. This is extremely unexpected, pleasefile a bug and make sure you are not manually modifying the SavedModel.')

    def _iter_all_nodes(self):
        for node_id in self._ordered_node_ids:
            yield (node_id, self._proto.nodes[node_id])

    def _load_nodes(self):
        """Load all saved objects."""
        nodes, node_setters = self._initialize_loaded_nodes()
        slot_variable_node_ids = {}
        for node_id, proto in self._iter_all_nodes():
            for slot_variable_proto in proto.slot_variables:
                slot_variable_node_id = slot_variable_proto.slot_variable_node_id
                slot_variable_node_ids[slot_variable_node_id] = (node_id, slot_variable_proto)
        for node_id, proto in self._iter_all_nodes():
            if nodes.get(node_id) is not None:
                continue
            elif node_id in slot_variable_node_ids:
                optimizer_node_id, slot_variable_proto = slot_variable_node_ids[node_id]
                optimizer_object = nodes[optimizer_node_id]
                optimized_variable = nodes[slot_variable_proto.original_variable_node_id]
                slot_variable = optimizer_object.add_slot(var=optimized_variable, slot_name=slot_variable_proto.slot_name)
                nodes[slot_variable_proto.slot_variable_node_id] = slot_variable
                node_setters[slot_variable_proto.slot_variable_node_id] = setattr
            else:
                node, setter = self._recreate(proto, node_id, nodes)
                nodes[node_id] = node
                node_setters[node_id] = setter
        if 0 not in nodes:
            nodes[0] = self._recreate_base_user_object()[0]
        self._nodes = [nodes.get(node_id) for node_id in range(len(self._proto.nodes))]
        self._node_setters = node_setters

    def _restore_checkpoint(self):
        """Load state from checkpoint into the deserialized objects."""
        variables_path = path_helpers.get_variables_path(self._export_dir)
        saver = checkpoint.TrackableSaver(graph_view.ObjectGraphView(self.get(0)))
        with ops.device('CPU'):
            saver._file_prefix_placeholder = constant_op.constant(variables_path)
        if self._save_options.allow_partial_checkpoint:
            load_status = saver.restore(variables_path, self._checkpoint_options).expect_partial()
            load_status.assert_nontrivial_match()
        else:
            load_status = saver.restore(variables_path, self._checkpoint_options)
            load_status.assert_existing_objects_matched()
        ckpt = load_status._checkpoint
        if not context.executing_eagerly():
            reader = py_checkpoint_reader.NewCheckpointReader(variables_path)
            for object_id, obj in dict(ckpt.object_by_proto_id).items():
                position = restore.CheckpointPosition(checkpoint=ckpt, proto_id=object_id)
                registered_saver = position.get_registered_saver_name()
                if registered_saver:
                    raise NotImplementedError(f'Loading a SavedModel that uses registered checkpoint saver is not supported in graph mode. The loaded object {obj} uses the saver registered with the name {registered_saver}.')
                restore_ops = position.restore_ops(reader)
                if restore_ops:
                    if resource_variable_ops.is_resource_variable(obj):
                        if len(restore_ops) == 1:
                            obj._initializer_op = restore_ops[0]
                        else:
                            obj._initializer_op = control_flow_ops.group(*restore_ops)
                    elif isinstance(obj, lookup_ops.LookupInterface) or isinstance(obj, resource.CapturableResource):
                        ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, restore_ops)
                    else:
                        raise NotImplementedError(f'Unable to restore state of object {obj} from the checkpoint.')

    def adjust_debug_info_func_names(self, debug_info):
        """Rewrite func names in the debug info by using the concrete func names."""
        output_debug_info = graph_debug_info_pb2.GraphDebugInfo()
        output_debug_info.files[:] = debug_info.files
        for key in debug_info.traces:
            node, func = key.split('@')
            new_func = ''
            if func in self._concrete_functions:
                new_func = self._concrete_functions[func].function_def.signature.name
            output_debug_info.traces[node + '@' + new_func].CopyFrom(debug_info.traces[key])
        return output_debug_info

    def get(self, node_id):
        if isinstance(node_id, str):
            node_id = self._node_path_to_id[node_id]
        return self._nodes[node_id]

    def _recreate(self, proto, node_id, nodes):
        """Creates a Python object from a SavedObject protocol buffer.

    Args:
      proto: a SavedObject proto
      node_id: int, the index of this object in the SavedObjectGraph node list.
      nodes: dict mapping int node_ids -> created objects.

    Returns:
      The recreated object, and the set-attribute function for reconnecting
      the trackable children.
    """
        registered_class = registration.get_registered_class(proto.registered_name)
        if registered_class is None:
            registered_class = _BUILT_IN_REGISTRATIONS.get(proto.WhichOneof('kind'))
        dependencies = {}
        for key, dep_node_id in self._get_node_dependencies(proto).items():
            dependencies[key] = nodes[dep_node_id]
        if registered_class:
            obj = registered_class._deserialize_from_proto(proto=proto.serialized_user_proto, object_proto=proto, dependencies=dependencies, export_dir=self._export_dir, asset_file_def=self._asset_file_def, operation_attributes=self._operation_attributes)
            if isinstance(obj, base.Trackable):
                setter = type(obj)._add_trackable_child
            else:
                setter = setattr
            return (obj, setter)
        else:
            return self._recreate_default(proto, node_id, dependencies)

    def _recreate_default(self, proto, node_id, deps):
        """Creates a Python object from a SavedObject protocol buffer."""
        factory = {'user_object': lambda: self._recreate_user_object(proto.user_object, node_id), 'function': lambda: self._recreate_function(proto.function, deps), 'bare_concrete_function': functools.partial(self._recreate_bare_concrete_function, proto=proto.bare_concrete_function, dependencies=deps), 'variable': lambda: self._recreate_variable(proto.variable), 'captured_tensor': functools.partial(self._get_tensor_from_fn, proto.captured_tensor)}
        kind = proto.WhichOneof('kind')
        if kind not in factory:
            raise ValueError(f'Unknown SavedObject type: {kind}. Expected one of {list(factory.keys())}.')
        return factory[kind]()

    def _recreate_user_object(self, proto, node_id):
        """Instantiates a SavedUserObject."""
        if proto.identifier == 'optimizer':
            try:
                import keras.optimizers.legacy as _
            except ImportError:
                try:
                    import keras.optimizers.optimizer_v2 as _
                except ImportError as e:
                    raise ImportError('Error when importing Keras. Unable to load SavedModel that contains an optimizer without the Keras module.') from e
        looked_up = revived_types.deserialize(proto)
        if looked_up is None:
            return self._recreate_base_user_object(proto, node_id)
        return looked_up

    def _recreate_base_user_object(self, proto=None, node_id=None):
        del proto, node_id

        class _UserObject(autotrackable.AutoTrackable):
            pass
        return (_UserObject(), setattr)

    def _recreate_function(self, proto, dependencies):
        fn = function_deserialization.recreate_function(proto, self._concrete_functions)
        for name in proto.concrete_functions:
            self._setup_function_captures(name, dependencies)
        if self._save_options.experimental_load_function_aliases:
            if proto.concrete_functions and all((name in self._concrete_function_aliases for name in proto.concrete_functions)):
                alias = self._concrete_function_aliases[next(iter(proto.concrete_functions))]
                aliased = self.function_aliases.get(alias)
                assert isinstance(aliased, list)
                if set((f.name for f in aliased)) == set((f.name for f in fn._list_all_concrete_functions())):
                    self.function_aliases[alias] = fn
                else:
                    logging.warn("Not aliasing '%s' to polymorphic restored function because of mismatched concrete functions: %s vs %s", alias, set((f.name for f in aliased)), set((f.name for f in fn._list_all_concrete_functions())))
        return (fn, setattr)

    def _recreate_bare_concrete_function(self, proto, dependencies):
        fn = function_deserialization.setup_bare_concrete_function(proto, self._concrete_functions)
        self._setup_function_captures(proto.concrete_function_name, dependencies)
        return (fn, setattr)

    def _recreate_variable(self, proto):
        name = proto.name if proto.name else None
        if name is not None:
            dbg_name = name
        else:
            dbg_name = '<variable loaded from saved model>'
        synchronization, aggregation, trainable = variables.validate_synchronization_aggregation_trainable(proto.synchronization, proto.aggregation, proto.trainable, name=dbg_name)

        def uninitialized_variable_creator(next_creator, **kwargs):
            """A variable creator that creates uninitialized variables."""
            del next_creator
            return resource_variable_ops.UninitializedVariable(**kwargs)
        with ops.get_default_graph()._variable_creator_scope(uninitialized_variable_creator, priority=50):
            saved_device = proto.device
            load_with_device = self._save_options.experimental_variable_policy._save_variable_devices() and config.get_soft_device_placement() and saved_device
            if load_with_device:
                with ops.device(saved_device):
                    return (variables.Variable(shape=proto.shape, dtype=proto.dtype, name=name, trainable=trainable, synchronization=synchronization, aggregation=aggregation), setattr)
            else:
                return (variables.Variable(shape=proto.shape, dtype=proto.dtype, name=name, trainable=trainable, synchronization=synchronization, aggregation=aggregation), setattr)

    def _get_tensor_from_fn(self, proto):
        outer_graph = self._concrete_functions[proto.concrete_function].graph
        captured_tensor = outer_graph.get_tensor_by_name(proto.name)
        return (captured_tensor, setattr)