from typing import List
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import type_spec
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _specs_for_flat_tensors(element_spec):
    """Return a flat list of type specs for element_spec.

  Note that "flat" in this function and in `_flat_tensor_specs` is a nickname
  for the "batchable tensor list" encoding used by datasets and map_fn
  internally (in C++/graphs). The ability to batch, unbatch and change
  batch size is one important characteristic of this encoding. A second
  important characteristic is that it represets a ragged tensor or sparse
  tensor as a single tensor of type variant (and this encoding uses special
  ops to encode/decode to/from variants).

  (In constrast, the more typical encoding, e.g. the C++/graph
  representation when calling a tf.function, is "component encoding" which
  represents sparse and ragged tensors as multiple dense tensors and does
  not use variants or special ops for encoding/decoding.)

  Args:
    element_spec: A nest of TypeSpec describing the elements of a dataset (or
      map_fn).

  Returns:
    A non-nested list of TypeSpec used by the encoding of tensors by
    datasets and map_fn for ELEMENT_SPEC. The items
    in this list correspond to the items in `_flat_tensor_specs`.
  """
    if isinstance(element_spec, StructuredTensor.Spec):
        specs = []
        for _, field_spec in sorted(element_spec._field_specs.items(), key=lambda t: t[0]):
            specs.extend(_specs_for_flat_tensors(field_spec))
    elif isinstance(element_spec, type_spec.BatchableTypeSpec) and element_spec.__class__._flat_tensor_specs is type_spec.BatchableTypeSpec._flat_tensor_specs:
        specs = nest.flatten(element_spec._component_specs, expand_composites=False)
    else:
        specs = nest.flatten(element_spec, expand_composites=False)
    return specs