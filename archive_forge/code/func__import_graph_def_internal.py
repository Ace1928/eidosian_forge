import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _import_graph_def_internal(graph_def, input_map=None, return_elements=None, validate_colocation_constraints=True, name=None, producer_op_list=None, propagate_device_spec=False):
    """Imports the graph from `graph_def` into the current default `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  `tf.Tensor` and `tf.Operation` objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  `tf.Graph.as_graph_def` for a way to create a `GraphDef`
  proto.

  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into the
      default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in `graph_def`
      that will be returned as `Operation` objects; and/or tensor names in
      `graph_def` that will be returned as `Tensor` objects.
    validate_colocation_constraints: Whether to validate colocation constraints.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.
    propagate_device_spec: Whether to propagate assigned device information
      when importing a graph from a GraphDef into the current default `Graph`.

  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`,
    and None if `returns_elements` is None.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
    graph_def = _ProcessGraphDefParam(graph_def)
    input_map = _ProcessInputMapParam(input_map)
    return_elements = _ProcessReturnElementsParam(return_elements)
    if producer_op_list is not None:
        _RemoveDefaultAttrs(producer_op_list, graph_def)
    graph = ops.get_default_graph()
    with ops.name_scope(name, 'import', input_map.values()) as scope:
        if scope:
            assert scope.endswith('/')
            prefix = scope[:-1]
        else:
            prefix = ''
        input_map = _ConvertInputMapValues(name, input_map)
    scoped_options = c_api_util.ScopedTFImportGraphDefOptions()
    options = scoped_options.options
    _PopulateTFImportGraphDefOptions(options, prefix, input_map, return_elements, validate_colocation_constraints, propagate_device_spec)
    with graph._mutation_lock():
        with c_api_util.tf_buffer(graph_def.SerializeToString()) as serialized:
            try:
                with graph._c_graph.get() as c_graph:
                    results = c_api.TF_GraphImportGraphDefWithResults(c_graph, serialized, options)
                results = c_api_util.ScopedTFImportGraphDefResults(results)
            except errors.InvalidArgumentError as e:
                raise ValueError(str(e))
        _ProcessNewOps(graph)
    if graph_def.library and graph_def.library.function:
        functions = function.from_library(graph_def.library)
        for f in functions:
            f.add_to_graph(graph)
    missing_unused_input_keys = c_api.TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(results.results)
    if missing_unused_input_keys:
        missing_unused_input_keys = [compat.as_str(s) for s in missing_unused_input_keys]
        missing_keys = ', '.join(missing_unused_input_keys)
        raise ValueError(f'Attempted to map inputs that were not found in graph_def: [{missing_keys}]')
    if return_elements is None:
        return None
    else:
        return _GatherReturnElements(return_elements, graph, results.results)