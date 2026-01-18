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
def _create_iterator(self, dataset):
    dataset = dataset._apply_debug_options()
    self._dataset = dataset
    ds_variant = dataset._variant_tensor
    self._element_spec = dataset.element_spec
    self._flat_output_types = structure.get_flat_tensor_types(self._element_spec)
    self._flat_output_shapes = structure.get_flat_tensor_shapes(self._element_spec)
    with ops.colocate_with(ds_variant):
        self._iterator_resource = gen_dataset_ops.anonymous_iterator_v3(output_types=self._flat_output_types, output_shapes=self._flat_output_shapes)
        if not context.executing_eagerly():
            fulltype = type_utils.iterator_full_type_from_spec(self._element_spec)
            assert len(fulltype.args[0].args[0].args) == len(self._flat_output_types)
            self._iterator_resource.op.experimental_set_type(fulltype)
        gen_dataset_ops.make_iterator(ds_variant, self._iterator_resource)