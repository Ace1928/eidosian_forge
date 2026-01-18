import abc
import threading
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.data.ops import iterator_autograph
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _next_internal(self):
    autograph_status = autograph_ctx.control_status_ctx().status
    autograph_disabled = autograph_status == autograph_ctx.Status.DISABLED
    if not context.executing_eagerly() and autograph_disabled:
        self._get_next_call_count += 1
        if self._get_next_call_count > GET_NEXT_CALL_ERROR_THRESHOLD:
            raise ValueError(GET_NEXT_CALL_ERROR_MESSAGE)
    if not context.executing_eagerly():
        with ops.colocate_with(self._iterator_resource):
            ret = gen_dataset_ops.iterator_get_next(self._iterator_resource, output_types=self._flat_output_types, output_shapes=self._flat_output_shapes)
        return structure.from_compatible_tensor_list(self._element_spec, ret)
    with context.execution_mode(context.SYNC):
        ret = gen_dataset_ops.iterator_get_next(self._iterator_resource, output_types=self._flat_output_types, output_shapes=self._flat_output_shapes)
        try:
            return self._element_spec._from_compatible_tensor_list(ret)
        except AttributeError:
            return structure.from_compatible_tensor_list(self._element_spec, ret)