import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def _create_tensor_maps(self):
    """Creates tensor to cache id maps."""
    self.tensorname_to_cache_idx = {}
    self.cache_idx_to_tensor_idx = []
    for out_tensor in self.traced_tensors:
        tensor_name = out_tensor.name
        if tensor_name in self.tensorname_to_cache_idx:
            raise ValueError('Tensor name {} should not be already in tensorname_to_cache_idx'.format(tensor_name))
        if tensor_name not in self.graph_order.tensor_to_idx:
            raise ValueError('Tensor name {} is not in the tensor_to_idx, tensor_to_idx={} '.format(tensor_name, self.graph_order.tensor_to_idx))
        tensor_idx = self.graph_order.tensor_to_idx[tensor_name]
        cache_idx = len(self.tensorname_to_cache_idx)
        self.tensorname_to_cache_idx[tensor_name] = cache_idx
        self.cache_idx_to_tensor_idx.append(tensor_idx)
        if len(self.tensorname_to_cache_idx) != len(self.cache_idx_to_tensor_idx):
            raise RuntimeError('len(self.tensorname_to_cache_idx) must equallen(self.cache_idx_to_tensor_idx), got len(self.tensorname_to_cache_idx)={}, len(self.cache_idx_to_tensor_idx)={}'.format(len(self.tensorname_to_cache_idx), len(self.cache_idx_to_tensor_idx)))