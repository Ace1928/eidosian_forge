from typing import List
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import type_spec
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _translate_to_fulltype_for_flat_tensors(spec: type_spec.TypeSpec) -> List[full_type_pb2.FullTypeDef]:
    """Convert a TypeSec to a list of FullTypeDef.

  The FullTypeDef created corresponds to the encoding used with datasets
  (and map_fn) that uses variants (and not FullTypeDef corresponding to the
  default "component" encoding).

  Currently, the only use of this is for information about the contents of
  ragged tensors, so only ragged tensors return useful full type information
  and other types return TFT_UNSET. While this could be improved in the future,
  this function is intended for temporary use and expected to be removed
  when type inference support is sufficient.

  Args:
    spec: A TypeSpec for one element of a dataset or map_fn.

  Returns:
    A list of FullTypeDef corresponding to SPEC. The length of this list
    is always the same as the length of spec._flat_tensor_specs.
  """
    if isinstance(spec, RaggedTensorSpec):
        dt = spec.dtype
        elem_t = _DT_TO_FT.get(dt)
        if elem_t is None:
            logging.vlog(1, 'dtype %s that has no conversion to fulltype.', dt)
        elif elem_t == full_type_pb2.TFT_LEGACY_VARIANT:
            logging.vlog(1, 'Ragged tensors containing variants are not supported.', dt)
        else:
            assert len(spec._flat_tensor_specs) == 1
            return [full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_RAGGED, args=[full_type_pb2.FullTypeDef(type_id=elem_t)])]
    return [full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET) for t in spec._flat_tensor_specs]