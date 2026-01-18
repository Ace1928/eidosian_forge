import collections
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import constants
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
class CheckpointPosition(object):
    """Indicates a position within a `_CheckpointRestoreCoordinator`."""
    __slots__ = ['_checkpoint', '_proto_id', 'skip_restore']

    def __init__(self, checkpoint, proto_id):
        """Specify an object within a checkpoint.

    Args:
      checkpoint: A _CheckpointRestoreCoordinator object.
      proto_id: The index of this object in TrackableObjectGraph.nodes.
    """
        self._checkpoint = checkpoint
        self._proto_id = proto_id
        self.skip_restore = False

    def restore(self, trackable, reader=None):
        """Restore this value into `trackable`."""
        with ops.init_scope():
            if self.bind_object(trackable):
                restore_ops = self._restore_descendants(reader)
                if restore_ops:
                    self._checkpoint.new_restore_ops(restore_ops)

    def bind_object(self, trackable):
        """Set a checkpoint<->object correspondence.

    Args:
      trackable: The object to record a correspondence for.

    Returns:
      True if this is a new assignment, False if this object has already been
      mapped to a checkpointed `Object` proto.
    Raises:
      AssertionError: If another object is already bound to the `Object` proto.
    """
        checkpoint = self.checkpoint
        checkpoint.all_python_objects.add(trackable)
        current_assignment = checkpoint.object_by_proto_id.get(self._proto_id, None)
        checkpoint.matched_proto_ids.add(self._proto_id)
        if current_assignment is None:
            checkpoint.object_by_proto_id[self._proto_id] = trackable
            return True
        else:
            if current_assignment is not trackable:
                logging.warning(f'Inconsistent references when loading the checkpoint into this object graph. For example, in the saved checkpoint object, `model.layer.weight` and `model.layer_copy.weight` reference the same variable, while in the current object these are two different variables. The referenced variables are:({current_assignment} and {trackable}).')
            return False

    def is_simple_variable(self):
        """Determine whether this value is restorable with a Tensor initializer."""
        attributes = self.object_proto.attributes
        return len(attributes) == 1 and attributes[0].name == constants.VARIABLE_VALUE_KEY and (not self.object_proto.children)

    def value_tensors(self, shape_and_slices=None):
        """Create value `Tensor`s for this object's attributes.

    Does not require that the Python object has been created. Used for
    restore-on-create when executing eagerly.

    Args:
      shape_and_slices: A dict mapping from object attribute names to a shape
        and slice string that will be passed to a RestoreV2 op. If the dict is
        None or if an object attribute is not in the dict, the full tensor will
        be restored.

    Returns:
      A dictionary mapping from object attribute names to `Tensor`s.
    """
        value_tensors = {}
        for serialized_tensor in self.object_proto.attributes:
            checkpoint_key = serialized_tensor.checkpoint_key
            dtype = self._checkpoint.dtype_map[checkpoint_key]
            base_type = dtype.base_dtype
            io_device = self._checkpoint.options.experimental_io_device or 'cpu:0'
            with ops.init_scope():
                with ops.device(io_device):
                    if shape_and_slices is not None and serialized_tensor.name in shape_and_slices:
                        shape_and_slice = shape_and_slices[serialized_tensor.name]
                    else:
                        shape_and_slice = ''
                    value, = io_ops.restore_v2(prefix=self._checkpoint.save_path_tensor, tensor_names=[checkpoint_key], shape_and_slices=[shape_and_slice], dtypes=[base_type], name='%s_checkpoint_read' % (serialized_tensor.name,))
                value_tensors[serialized_tensor.name] = array_ops.identity(value)
        return value_tensors

    def gather_ops_or_named_saveables(self):
        """Looks up or creates SaveableObjects which don't have cached ops.

    Returns:
      A tuple of (
          existing_restore_ops: list,
          named_saveables: dict,
          python_positions: list,
          registered_savers: dict)
    """
        recorded_registered_saver = self.get_registered_saver_name()
        if not (self.object_proto.attributes or recorded_registered_saver):
            return ([], {}, [], {})
        existing_restore_ops = []
        named_saveables = {}
        python_positions = []
        registered_savers = collections.defaultdict(dict)
        saveable_factories = saveable_object_util.saveable_objects_from_trackable(self.trackable)
        saver_name = registration.get_registered_saver_name(self.trackable)
        if recorded_registered_saver:
            if not self.skip_restore:
                name = self.object_proto.registered_saver.object_name
                registered_savers[recorded_registered_saver][name] = self.trackable
        elif saver_name:
            registered_savers[saver_name] = {self.object_proto.attributes[0].checkpoint_key: self.trackable}
        elif isinstance(self.trackable, python_state.PythonState):
            python_positions.append(self)
        elif saveable_factories.keys() == {trackable_utils.SERIALIZE_TO_TENSORS_NAME}:
            existing_restore_ops, named_saveables = self._create_serialize_to_tensor_saveable(saveable_factories)
        elif saveable_factories:
            existing_restore_ops, named_saveables = self._create_saveables_by_attribute_name(saveable_factories)
        else:
            for serialized_tensor in self.object_proto.attributes:
                self._checkpoint.unused_attributes.setdefault(self._proto_id, []).append(serialized_tensor.name)
        return (existing_restore_ops, named_saveables, python_positions, registered_savers)

    def _create_serialize_to_tensor_saveable(self, saveable_factories):
        """Creates a saveable using the _serialize_to_tensor method."""
        suffix = saveable_compat.get_saveable_name(self.trackable) or ''
        saveable_name = _extract_saveable_name(self.object_proto.attributes[0].checkpoint_key) + suffix
        if not context.executing_eagerly():
            existing_op = self._checkpoint.restore_ops_by_name.get(saveable_name, None)
            if existing_op is not None:
                return ([existing_op], {})
            saveables_cache = self._checkpoint.saveables_cache.setdefault(self.trackable, {})
            if saveable_name in saveables_cache:
                return ([], {saveable_name: saveables_cache[saveable_name]})
        saveable = saveable_factories[trackable_utils.SERIALIZE_TO_TENSORS_NAME](name=saveable_name)
        if not context.executing_eagerly():
            saveables_cache[saveable_name] = saveable
        return ([], {saveable_name: saveable})

    def _create_saveables_by_attribute_name(self, saveable_factories):
        """Creates or caches SaveableObjects by matching the attribute names.

    The attribute name keys in the `saveable_factories` is used to find the
    corresponding attribute in the object proto. Attributes contain checkpoint
    keys which are passed to the factory function to generate the
    SaveableObject.

    Args:
      saveable_factories: a dict mapping attribute name to a callable factory
        function that produces a SaveableObject.

    Returns:
      A tuple of (
          existing_restore_ops: list,
          named_saveables: dict)
    """
        named_saveables = {}
        existing_restore_ops = []
        created_compat_names = set()
        for serialized_tensor in self.object_proto.attributes:
            if context.executing_eagerly():
                existing_op = None
            else:
                existing_op = self._checkpoint.restore_ops_by_name.get(serialized_tensor.checkpoint_key, None)
            if existing_op is not None:
                existing_restore_ops.append(existing_op)
                continue
            if any((serialized_tensor.name.startswith(name) for name in created_compat_names)):
                continue
            saveables_cache = self._checkpoint.saveables_cache
            if saveables_cache is None:
                saveable = None
            else:
                saveable_list = saveables_cache.get(self.trackable, {}).get(serialized_tensor.name, (None,))
                if len(saveable_list) == 1:
                    saveable, = saveable_list
                else:
                    saveable = None
            if saveable is not None:
                if serialized_tensor.checkpoint_key not in saveable.name:
                    saveable = None
                    del saveables_cache[self.trackable]
            if saveable is None:
                saveable = _get_saveable_from_factory(saveable_factories, serialized_tensor, created_compat_names)
                if saveable is None:
                    self._checkpoint.unused_attributes.setdefault(self._proto_id, []).append(serialized_tensor.name)
                    continue
                if saveables_cache is not None:
                    saveables_cache.setdefault(self.trackable, {})[serialized_tensor.name] = [saveable]
            named_saveables[serialized_tensor.checkpoint_key] = saveable
        return (existing_restore_ops, named_saveables)

    def restore_ops(self, reader=None):
        """Create or fetch restore ops for this object's attributes.

    Requires that the `Trackable` Python object has been bound to an object
    ID in the checkpoint.

    Args:
      reader: A `CheckpointReader`. If None, a new instance will be created.

    Returns:
      A list of operations when graph building, or an empty list when executing
      eagerly.
    """
        if self._has_registered_saver():
            raise ValueError('Unable to run individual checkpoint restore for objects with registered savers.')
        restore_ops, tensor_saveables, python_positions, _ = self.gather_ops_or_named_saveables()
        restore_ops.extend(self._checkpoint.restore_saveables(tensor_saveables, python_positions, reader=reader))
        return restore_ops

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def trackable(self):
        return self._checkpoint.object_by_proto_id[self._proto_id]

    @property
    def object_proto(self):
        return self._checkpoint.object_graph_proto.nodes[self._proto_id]

    @property
    def proto_id(self):
        return self._proto_id

    @property
    def restore_uid(self):
        return self._checkpoint.restore_uid

    def __repr__(self):
        return repr(self.object_proto)

    def value_shape(self):
        """The shape of the VARIABLE_VALUE tensor.

    Returns:
      If found a TensorShape object, otherwise None.
    """
        for serialized_tensor in self.object_proto.attributes:
            if serialized_tensor.name == constants.VARIABLE_VALUE_KEY:
                return self._checkpoint.shape_map[serialized_tensor.checkpoint_key]
        return None

    def _has_registered_saver(self):
        return bool(self.object_proto.registered_saver.name)

    def get_registered_saver_name(self):
        """Returns the registered saver name defined in the Checkpoint."""
        if self._has_registered_saver():
            saver_name = self.object_proto.registered_saver.name
            try:
                registration.validate_restore_function(self.trackable, saver_name)
            except ValueError as e:
                if registration.get_strict_predicate_restore(saver_name):
                    raise e
                self.skip_restore = True
            return saver_name
        return None

    def create_slot_variable_position(self, optimizer_object, variable, slot_variable_id, slot_name):
        """Generates CheckpointPosition for a slot variable.

    Args:
      optimizer_object: Optimizer that owns the slot variable.
      variable: Variable associated with the slot variable.
      slot_variable_id: ID of the slot variable.
      slot_name: Name of the slot variable.

    Returns:
      If there is a slot variable in the `optimizer_object` that has not been
      bound to the checkpoint, this function returns a tuple of (
        new `CheckpointPosition` for the slot variable,
        the slot variable itself).
    """
        slot_variable_position = CheckpointPosition(checkpoint=self.checkpoint, proto_id=slot_variable_id)
        slot_variable = optimizer_object._create_or_restore_slot_variable(slot_variable_position=slot_variable_position, variable=variable, slot_name=slot_name)
        if slot_variable is not None and slot_variable_position.bind_object(slot_variable):
            return (slot_variable_position, slot_variable)
        else:
            return (None, None)

    def create_child_position(self, node_id):
        return CheckpointPosition(checkpoint=self.checkpoint, proto_id=node_id)

    def _restore_descendants(self, reader=None):
        """Restore the bound Trackable and dependencies (may be deferred)."""
        visit_queue = collections.deque([(self, self.trackable)])
        restore_ops = []
        tensor_saveables = {}
        python_positions = []
        registered_savers = collections.defaultdict(dict)
        while visit_queue:
            current_position, _ = visit_queue.popleft()
            new_restore_ops, new_tensor_saveables, new_python_positions, new_registered_savers = current_position._single_restore()
            restore_ops.extend(new_restore_ops)
            tensor_saveables.update(new_tensor_saveables)
            python_positions.extend(new_python_positions)
            for saver_name, trackable_map in new_registered_savers.items():
                registered_savers[saver_name].update(trackable_map)
            _queue_children_for_restoration(current_position, visit_queue)
            _queue_slot_variables(current_position, visit_queue)
        restore_ops.extend(current_position.checkpoint.restore_saveables(tensor_saveables, python_positions, registered_savers, reader=reader))
        return restore_ops

    def _single_restore(self):
        """Restores the trackable."""
        trackable = self.trackable
        trackable._maybe_initialize_trackable()
        checkpoint = self.checkpoint
        if checkpoint.restore_uid > trackable._update_uid:
            restore_ops, tensor_saveables, python_positions, registered_savers = self.gather_ops_or_named_saveables()
            trackable._update_uid = checkpoint.restore_uid
        else:
            restore_ops = ()
            tensor_saveables = {}
            python_positions = ()
            registered_savers = {}
        return (restore_ops, tensor_saveables, python_positions, registered_savers)