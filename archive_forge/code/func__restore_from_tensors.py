from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.checkpoint import saveable_compat
def _restore_from_tensors(self, restored_tensors):
    with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_import_v2(self.resource_handle, restored_tensors['-keys'], restored_tensors['-values'])