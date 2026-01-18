import collections
import copy
import itertools
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class Functional(training_lib.Model):
    """A `Functional` model is a `Model` defined as a directed graph of layers.

  Three types of `Model` exist: subclassed `Model`, `Functional` model,
  and `Sequential` (a special case of `Functional`).
  In general, more Keras features are supported with `Functional`
  than with subclassed `Model`s, specifically:

  - Model cloning (`keras.models.clone`)
  - Serialization (`model.get_config()/from_config`, `model.to_json()`
  - Whole-model saving (`model.save()`)

  A `Functional` model can be instantiated by passing two arguments to
  `__init__`. The first argument is the `keras.Input` Tensors that represent
  the inputs to the model. The second argument specifies the output
  tensors that represent the outputs of this model. Both arguments can be a
  nested structure of tensors.

  Example:

  ```
  inputs = {'x1': keras.Input(shape=(10,)), 'x2': keras.Input(shape=(1,))}
  t = keras.layers.Dense(1, activation='relu')(inputs['x1'])
  outputs = keras.layers.Add()([t, inputs['x2'])
  model = keras.Model(inputs, outputs)
  ```

  A `Functional` model constructed using the Functional API can also include raw
  TensorFlow functions, with the exception of functions that create Variables
  or assign ops.

  Example:

  ```
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(1)(inputs)
  outputs = tf.nn.relu(x)
  model = keras.Model(inputs, outputs)
  ```

  Args:
    inputs: List of input tensors (must be created via `tf.keras.Input()`).
    outputs: List of output tensors.
    name: String, optional. Name of the model.
    trainable: Boolean, optional. If the model's variables should be trainable.
  """
    _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(('_layer_call_argspecs', '_compiled_trainable_state', '_output_mask_cache', '_output_tensor_cache', '_output_shape_cache'), training_lib.Model._TF_MODULE_IGNORED_PROPERTIES))

    @trackable.no_automatic_dependency_tracking
    def __init__(self, inputs, outputs, name=None, trainable=True, **kwargs):
        skip_init = kwargs.pop('skip_init', False)
        if skip_init:
            return
        generic_utils.validate_kwargs(kwargs, {})
        super(Functional, self).__init__(name=name, trainable=trainable)
        self._init_graph_network(inputs, outputs)

    @trackable.no_automatic_dependency_tracking
    def _init_graph_network(self, inputs, outputs):
        self._is_graph_network = True
        if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
            inputs = inputs[0]
        if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
            outputs = outputs[0]
        self._nested_inputs = inputs
        self._nested_outputs = outputs
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        if not nest.is_nested(self._nested_inputs):
            self._enable_dict_to_input_mapping = True
        elif isinstance(self._nested_inputs, (list, tuple)) and (not any((nest.is_nested(t) for t in self._nested_inputs))):
            self._enable_dict_to_input_mapping = True
        elif isinstance(self._nested_inputs, dict) and (not any((nest.is_nested(t) for t in self._nested_inputs.values()))):
            self._enable_dict_to_input_mapping = True
        else:
            self._enable_dict_to_input_mapping = False
        if not ops.executing_eagerly_outside_functions():
            if any((not hasattr(tensor, '_keras_history') for tensor in self.outputs)):
                base_layer_utils.create_keras_history(self._nested_outputs)
        self._validate_graph_inputs_and_outputs()
        self.built = True
        self._build_input_shape = nest.map_structure(lambda x: x.shape, inputs)
        self._compute_output_and_mask_jointly = True
        self._expects_training_arg = True
        self._expects_mask_arg = True
        self._autocast = False
        self._input_layers = []
        self._output_layers = []
        self._input_coordinates = []
        self._output_coordinates = []
        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            self._output_layers.append(layer)
            self._output_coordinates.append((layer, node_index, tensor_index))
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            assert node_index == 0
            assert tensor_index == 0
            self._input_layers.append(layer)
            self._input_coordinates.append((layer, node_index, tensor_index))
        nodes, nodes_by_depth, layers, _ = _map_graph_network(self.inputs, self.outputs)
        self._network_nodes = nodes
        self._nodes_by_depth = nodes_by_depth
        self._self_tracked_trackables = layers
        self._layer_call_argspecs = {}
        for layer in self._self_tracked_trackables:
            self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
        self._set_output_names()
        self.input_names = []
        self._feed_input_names = []
        self._feed_inputs = []
        self._feed_input_shapes = []
        for layer in self._input_layers:
            self.input_names.append(layer.name)
            if layer.is_placeholder:
                self._feed_input_names.append(layer.name)
                self._feed_input_shapes.append(layer._batch_input_shape)
                self._feed_inputs.append(layer.input)
        self._compute_tensor_usage_count()
        self._set_save_spec(self._nested_inputs)
        tf_utils.assert_no_legacy_layers(self.layers)

    @property
    def input(self):
        """Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    """
        return self._nested_inputs

    @property
    def input_shape(self):
        """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """
        return nest.map_structure(backend.int_shape, self.input)

    @property
    def input_spec(self):
        if hasattr(self, '_manual_input_spec'):
            return self._manual_input_spec
        if isinstance(self._nested_inputs, (dict, list, tuple)) and len(self._nested_inputs) != len(self.inputs):
            return None
        if isinstance(self._nested_inputs, dict):
            names = sorted(self._nested_inputs.keys())
            return [input_spec.InputSpec(shape=shape_with_no_batch_size(self._nested_inputs[name]), allow_last_axis_squeeze=True, name=name) for name in names]
        else:
            return [input_spec.InputSpec(shape=shape_with_no_batch_size(x), allow_last_axis_squeeze=True, name=x._keras_history.layer.name) for x in self.inputs]

    @input_spec.setter
    def input_spec(self, value):
        self._manual_input_spec = value

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
      Output tensor or list of output tensors.

    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    """
        return self._nested_outputs

    @property
    def output_shape(self):
        """Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    """
        return nest.map_structure(backend.int_shape, self.output)

    def _set_output_names(self):
        """Assigns unique names to the Network's outputs.

    Output layers with multiple output tensors would otherwise lead to duplicate
    names in self.output_names.
    """
        uniquified = []
        output_names = set()
        prefix_count = {}
        for layer in self._output_layers:
            proposal = layer.name
            while proposal in output_names:
                existing_count = prefix_count.get(layer.name, 1)
                proposal = '{}_{}'.format(layer.name, existing_count)
                prefix_count[layer.name] = existing_count + 1
            output_names.add(proposal)
            uniquified.append(proposal)
        self.output_names = uniquified

    @property
    def _layer_checkpoint_dependencies(self):
        """Dictionary of layer dependencies to be included in the checkpoint."""
        weight_layer_index = 0
        dependencies = collections.OrderedDict()
        for layer_index, layer in enumerate(self.layers):
            try:
                if layer.weights:
                    dependencies['layer_with_weights-%d' % weight_layer_index] = layer
                    weight_layer_index += 1
            except ValueError:
                pass
            dependencies['layer-%d' % layer_index] = layer
        return dependencies

    def _trackable_children(self, save_type=trackable.SaveType.CHECKPOINT, **kwargs):
        dependencies = self._layer_checkpoint_dependencies
        dependencies.update(super(Functional, self)._trackable_children(save_type, **kwargs))
        return dependencies

    def _lookup_dependency(self, name):
        layer_dependencies = self._layer_checkpoint_dependencies
        if name in layer_dependencies:
            return layer_dependencies[name]
        return super(Functional, self)._lookup_dependency(name)

    def _handle_deferred_layer_dependencies(self, layers):
        """Handles layer checkpoint dependencies that are added after init."""
        layer_checkpoint_dependencies = self._layer_checkpoint_dependencies
        layer_to_name = {v: k for k, v in layer_checkpoint_dependencies.items()}
        for layer in layers:
            if layer in layer_to_name:
                self._handle_deferred_dependencies(name=layer_to_name[layer], trackable=layer)

    @property
    def _should_compute_mask(self):
        return True

    def compute_mask(self, inputs, mask):
        output_tensors = self._run_internal_graph(inputs, mask=mask)
        return nest.map_structure(lambda t: getattr(t, '_keras_mask', None), output_tensors)

    @doc_controls.do_not_doc_inheritable
    def call(self, inputs, training=None, mask=None):
        """Calls the model on new inputs.

    In this case `call` just reapplies
    all ops in the graph to the new inputs
    (e.g. build a new computational graph from the provided inputs).

    Args:
        inputs: A tensor or list of tensors.
        training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

    Returns:
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
    """
        return self._run_internal_graph(inputs, training=training, mask=mask)

    def compute_output_shape(self, input_shape):
        input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
        if len(nest.flatten(input_shape)) != len(nest.flatten(self._input_layers)):
            raise ValueError('Invalid input_shape argument ' + str(input_shape) + ': model has ' + str(len(self._input_layers)) + ' tensor inputs.')
        try:
            cache_key = tuple(tf_utils.convert_shapes(input_shape, to_tuples=True))
            if cache_key in self._output_shape_cache:
                return self._output_shape_cache[cache_key]
        except ValueError:
            pass
        layers_to_output_shapes = {}
        for layer, shape in zip(self._input_layers, nest.flatten(input_shape)):
            shape_key = layer.name + '_0_0'
            layers_to_output_shapes[shape_key] = shape
        depth_keys = list(self._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        if len(depth_keys) > 1:
            for depth in depth_keys:
                nodes = self._nodes_by_depth[depth]
                for node in nodes:
                    layer = node.layer
                    if layer in self._input_layers:
                        continue
                    layer_input_shapes = []
                    layer_inputs = node.call_args[0]
                    for layer_input in nest.flatten(layer_inputs):
                        kh = layer_input._keras_history
                        input_layer_key = kh.layer.name + '_%s_%s' % (kh.node_index, kh.tensor_index)
                        layer_input_shapes.append(layers_to_output_shapes[input_layer_key])
                    layer_input_shapes = nest.pack_sequence_as(layer_inputs, layer_input_shapes)
                    layer_input_shapes = tf_utils.convert_shapes(layer_input_shapes, to_tuples=True)
                    layer_output_shapes = layer.compute_output_shape(layer_input_shapes)
                    layer_output_shapes = tf_utils.convert_shapes(layer_output_shapes, to_tuples=False)
                    node_index = layer._inbound_nodes.index(node)
                    for j, shape in enumerate(nest.flatten(layer_output_shapes)):
                        shape_key = layer.name + '_%s_%s' % (node_index, j)
                        layers_to_output_shapes[shape_key] = shape
            output_shapes = []
            for i in range(len(self._output_layers)):
                layer, node_index, tensor_index = self._output_coordinates[i]
                shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
                output_shapes.append(layers_to_output_shapes[shape_key])
            output_shapes = nest.pack_sequence_as(self._nested_outputs, output_shapes)
            self._output_shape_cache[cache_key] = output_shapes
        return output_shapes

    def _init_set_name(self, name, zero_based=True):
        if not name:
            cls_name = self.__class__.__name__
            if self.__class__ == Functional:
                cls_name = 'Model'
            self._name = backend.unique_object_name(generic_utils.to_snake_case(cls_name), zero_based=zero_based)
        else:
            self._name = name

    def _run_internal_graph(self, inputs, training=None, mask=None):
        """Computes output tensors for new inputs.

    # Note:
        - Can be run on non-Keras tensors.

    Args:
        inputs: Tensor or nested structure of Tensors.
        training: Boolean learning phase.
        mask: (Optional) Tensor or nested structure of Tensors.

    Returns:
        output_tensors
    """
        inputs = self._flatten_to_reference_inputs(inputs)
        if mask is None:
            masks = [None] * len(inputs)
        else:
            masks = self._flatten_to_reference_inputs(mask)
        for input_t, mask in zip(inputs, masks):
            input_t._keras_mask = mask
        tensor_dict = {}
        tensor_usage_count = self._tensor_usage_count
        for x, y in zip(self.inputs, inputs):
            y = self._conform_to_reference_input(y, ref_input=x)
            x_id = str(id(x))
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
        nodes_by_depth = self._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            for node in nodes:
                if node.is_input:
                    continue
                if any((t_id not in tensor_dict for t_id in node.flat_input_ids)):
                    continue
                args, kwargs = node.map_arguments(tensor_dict)
                outputs = node.layer(*args, **kwargs)
                for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
                    tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
        output_tensors = []
        for x in self.outputs:
            x_id = str(id(x))
            assert x_id in tensor_dict, 'Could not compute output ' + str(x)
            output_tensors.append(tensor_dict[x_id].pop())
        return nest.pack_sequence_as(self._nested_outputs, output_tensors)

    def _flatten_to_reference_inputs(self, tensors):
        """Maps `tensors` to their respective `keras.Input`."""
        if self._enable_dict_to_input_mapping and isinstance(tensors, dict):
            ref_inputs = self._nested_inputs
            if not nest.is_nested(ref_inputs):
                ref_inputs = [self._nested_inputs]
            if isinstance(ref_inputs, dict):
                ref_input_names = sorted(ref_inputs.keys())
            else:
                ref_input_names = [inp._keras_history.layer.name for inp in ref_inputs]
            if len(tensors) > len(ref_input_names):
                warnings.warn('Input dict contained keys {} which did not match any model input. They will be ignored by the model.'.format([n for n in tensors.keys() if n not in ref_input_names]))
            try:
                return [tensors[n] for n in ref_input_names]
            except KeyError:
                return nest.flatten(tensors)
        return nest.flatten(tensors)

    def _conform_to_reference_input(self, tensor, ref_input):
        """Set shape and dtype based on `keras.Input`s."""
        if isinstance(tensor, tensor_lib.Tensor):
            t_shape = tensor.shape
            t_rank = t_shape.rank
            ref_shape = ref_input.shape
            ref_rank = ref_shape.rank
            keras_history = getattr(tensor, '_keras_history', None)
            if t_rank is not None and ref_rank is not None:
                if t_rank == ref_rank + 1 and t_shape[-1] == 1:
                    tensor = array_ops.squeeze_v2(tensor, axis=-1)
                elif t_rank == ref_rank - 1 and ref_shape[-1] == 1:
                    tensor = array_ops.expand_dims_v2(tensor, axis=-1)
            if keras_history is not None:
                tensor._keras_history = keras_history
            if not context.executing_eagerly():
                try:
                    tensor.set_shape(tensor.shape.merge_with(ref_input.shape))
                except ValueError:
                    logging.warning('Model was constructed with shape {} for input {}, but it was called on an input with incompatible shape {}.'.format(ref_input.shape, ref_input, tensor.shape))
            tensor = math_ops.cast(tensor, dtype=ref_input.dtype)
        elif tf_utils.is_extension_type(tensor):
            ref_input_dtype = getattr(ref_input, 'dtype', None)
            if ref_input_dtype is not None and ref_input_dtype != dtypes.variant:
                tensor = math_ops.cast(tensor, dtype=ref_input_dtype)
        return tensor

    def get_config(self):
        return copy.deepcopy(get_network_config(self))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Instantiates a Model from its config (output of `get_config()`).

    Args:
        config: Model config dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A model instance.

    Raises:
        ValueError: In case of improperly formatted config dict.
    """
        with generic_utils.SharedObjectLoadingScope():
            input_tensors, output_tensors, created_layers = reconstruct_from_config(config, custom_objects)
            model = cls(inputs=input_tensors, outputs=output_tensors, name=config.get('name'))
            connect_ancillary_layers(model, created_layers)
            return model

    def _validate_graph_inputs_and_outputs(self):
        """Validates the inputs and outputs of a Graph Network."""
        if len({id(i) for i in self.inputs}) != len(self.inputs):
            raise ValueError('The list of inputs passed to the model is redundant. All inputs should only appear once. Found: ' + str(self.inputs))
        for x in self.inputs:
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise ValueError('Input tensors to a ' + cls_name + ' ' + 'must come from `tf.keras.Input`. Received: ' + str(x) + ' (missing previous layer metadata).')
            layer = x._keras_history.layer
            if len(layer._inbound_nodes) > 1 or (layer._inbound_nodes and (not layer._inbound_nodes[0].is_input)):
                cls_name = self.__class__.__name__
                logging.warning(cls_name + ' model inputs must come from `tf.keras.Input` (thus holding past layer metadata), they cannot be the output of a previous non-Input layer. Here, a tensor specified as input to "' + self.name + '" was not an Input tensor, it was generated by layer ' + layer.name + '.\nNote that input tensors are instantiated via `tensor = tf.keras.Input(shape)`.\nThe tensor that caused the issue was: ' + str(x.name))
        input_batch_sizes = [training_utils.get_static_batch_size(x._keras_history.layer) for x in self.inputs]
        consistent_batch_size = None
        for batch_size in input_batch_sizes:
            if batch_size is not None:
                if consistent_batch_size is not None and batch_size != consistent_batch_size:
                    raise ValueError('The specified batch sizes of the Input Layers are incompatible. Found batch sizes: {}'.format(input_batch_sizes))
                consistent_batch_size = batch_size
        for x in self.outputs:
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise ValueError('Output tensors of a ' + cls_name + ' model must be the output of a TensorFlow `Layer` (thus holding past layer metadata). Found: ' + str(x))

    def _insert_layers(self, layers, relevant_nodes=None):
        """Inserts Layers into the Network after Network creation.

    This is only valid for Keras Graph Networks.  Layers added via this function
    will be included in the `call` computation and `get_config` of this Network.
    They will not be added to the Network's outputs.


    Args:
      layers: Arbitrary nested structure of Layers. Layers must be reachable
        from one or more of the `keras.Input` Tensors that correspond to this
        Network's inputs.
      relevant_nodes: Nodes from the Layers that should be considered part of
        this Network. If `None`, all Nodes will be considered part of this
        Network.

    Raises:
      ValueError: If the layers depend on `Input`s not found in this Model.
    """
        layers = nest.flatten(layers)
        tf_utils.assert_no_legacy_layers(layers)
        node_to_depth = {}
        for depth, nodes in self._nodes_by_depth.items():
            node_to_depth.update({node: depth for node in nodes})
        if not relevant_nodes:
            relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
        network_nodes = set(relevant_nodes + list(node_to_depth.keys()))

        def _get_min_depth(node):
            """Gets the minimum depth at which node can be computed."""
            min_depth = 0
            for layer, node_id, _, _ in node.iterate_inbound():
                inbound_node = layer._inbound_nodes[node_id]
                if inbound_node in node_to_depth:
                    min_depth = min(min_depth, node_to_depth[inbound_node])
                elif inbound_node not in network_nodes:
                    continue
                else:
                    return None
            return min_depth - 1
        unprocessed_nodes = copy.copy(relevant_nodes)
        i = 0
        while unprocessed_nodes:
            i += 1
            if i > 10000:
                raise ValueError('Layers could not be added due to missing dependencies.')
            node = unprocessed_nodes.pop(0)
            depth = _get_min_depth(node)
            if depth is None:
                unprocessed_nodes.append(node)
                continue
            node_key = _make_node_key(node.layer.name, node.layer._inbound_nodes.index(node))
            if node_key not in self._network_nodes:
                node_to_depth[node] = depth
                self._network_nodes.add(node_key)
                self._nodes_by_depth[depth].append(node)
        layer_set = set(self._self_tracked_trackables)
        deferred_layers = []
        for layer in layers:
            if layer not in layer_set:
                self._self_tracked_trackables.append(layer)
                deferred_layers.append(layer)
                self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
                layer_set.add(layer)
        self._handle_deferred_layer_dependencies(deferred_layers)
        self._compute_tensor_usage_count()

    def _compute_tensor_usage_count(self):
        """Compute the #. of tensor usages for all the output tensors of layers.

    The computed tensor usage count is saved as `self._tensor_usage_count`. This
    is later used for saving memory in eager computation by releasing
    no-longer-needed tensors as early as possible.
    """
        tensor_usage_count = collections.Counter()
        available_tensors = set((str(id(tensor)) for tensor in self.inputs))
        depth_keys = list(self._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        depth_keys = depth_keys[1:]
        for depth in depth_keys:
            for node in self._nodes_by_depth[depth]:
                input_tensors = {str(id(tensor)) for tensor in nest.flatten(node.keras_inputs)}
                if input_tensors.issubset(available_tensors):
                    for tensor in nest.flatten(node.keras_inputs):
                        tensor_usage_count[str(id(tensor))] += 1
                    for output_tensor in nest.flatten(node.outputs):
                        available_tensors.add(str(id(output_tensor)))
        for tensor in self.outputs:
            tensor_usage_count[str(id(tensor))] += 1
        self._tensor_usage_count = tensor_usage_count

    def _assert_weights_created(self):
        return

    def _graph_network_add_loss(self, symbolic_loss):
        new_nodes, new_layers = _map_subgraph_network(self.inputs, [symbolic_loss])
        add_loss_layer = base_layer.AddLoss(unconditional=False, dtype=symbolic_loss.dtype)
        add_loss_layer(symbolic_loss)
        new_nodes.extend(add_loss_layer.inbound_nodes)
        new_layers.append(add_loss_layer)
        self._insert_layers(new_layers, new_nodes)

    def _graph_network_add_metric(self, value, aggregation, name):
        new_nodes, new_layers = _map_subgraph_network(self.inputs, [value])
        add_metric_layer = base_layer.AddMetric(aggregation, name, dtype=value.dtype)
        add_metric_layer(value)
        new_nodes.extend(add_metric_layer.inbound_nodes)
        new_layers.append(add_metric_layer)
        self._insert_layers(new_layers, new_nodes)

    @property
    def _trackable_saved_model_saver(self):
        return network_serialization.NetworkSavedModelSaver(self)

    def _get_save_spec(self, dynamic_batch=True):
        if getattr(self, '_has_explicit_input_shape', True):
            dynamic_batch = False
        return super(Functional, self)._get_save_spec(dynamic_batch)