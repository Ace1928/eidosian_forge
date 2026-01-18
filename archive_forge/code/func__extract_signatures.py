import functools
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import nest
def _extract_signatures(self, wrapped, meta_graph_def):
    """Creates ConcreteFunctions for signatures in `meta_graph_def`."""
    signature_functions = {}
    for signature_key, signature_def in meta_graph_def.signature_def.items():
        if signature_def.inputs:
            input_items = sorted(signature_def.inputs.items(), key=lambda item: item[0])
            original_input_names, input_specs = zip(*input_items)
        else:
            original_input_names = []
            input_specs = []
        feeds = [wrap_function._get_element_from_tensor_info(input_spec, wrapped.graph) for input_spec in input_specs]
        input_names = []
        input_tensors = []
        for original_input_name, feed in zip(original_input_names, feeds):
            if isinstance(feed, sparse_tensor.SparseTensor):
                indices_name = '%s_indices' % original_input_name
                values_name = '%s_values' % original_input_name
                dense_shape_name = '%s_dense_shape' % original_input_name
                input_names.extend([indices_name, values_name, dense_shape_name])
                input_tensors.extend([feed.indices, feed.values, feed.dense_shape])
            elif isinstance(feed, composite_tensor.CompositeTensor):
                component_tensors = nest.flatten(feed, expand_composites=True)
                input_names.extend(('%s_component_%d' % (original_input_name, n) for n in range(len(component_tensors))))
                input_tensors.extend(component_tensors)
            else:
                input_names.append(original_input_name)
                input_tensors.append(feed)
        fetches = {name: out for name, out in signature_def.outputs.items()}
        try:
            signature_fn = wrapped.prune(feeds=feeds, fetches=fetches)
        except lift_to_graph.UnliftableError as ex:
            args = ex.args
            if not args:
                message = ''
            else:
                message = args[0]
            message = "A SavedModel signature needs an input for each placeholder the signature's outputs use. An output for signature '{}' depends on a placeholder which is not an input (i.e. the placeholder is not fed a value).\n\n".format(signature_key) + message
            ex.args = (message,) + args[1:]
            raise
        signature_fn._arg_keywords = input_names
        signature_fn._func_graph.structured_input_signature = ((), func_graph.convert_structure_to_signature(dict(zip(input_names, input_tensors))))
        if len(input_names) == 1:
            signature_fn._num_positional_args = 1
        else:
            signature_fn._num_positional_args = 0
        signature_functions[signature_key] = signature_fn
    return signature_functions