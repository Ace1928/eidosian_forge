import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
class _ResourceSummaryWriter(SummaryWriter):
    """Implementation of SummaryWriter using a SummaryWriterInterface resource."""

    def __init__(self, create_fn, init_op_fn, mesh=None):
        if mesh is not None:
            with dtensor_api.default_mesh(mesh.host_mesh()):
                self._resource = create_fn()
                self._init_op = init_op_fn(self._resource)
        else:
            self._resource = create_fn()
            self._init_op = init_op_fn(self._resource)
        self._closed = False
        if context.executing_eagerly():
            self._set_up_resource_deleter()
        else:
            ops.add_to_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME, self._init_op)
        self._mesh = mesh

    def _set_up_resource_deleter(self):
        self._resource_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._resource, handle_device='cpu:0')

    def set_as_default(self, step=None):
        """See `SummaryWriter.set_as_default`."""
        if context.executing_eagerly() and self._closed:
            raise RuntimeError(f'SummaryWriter {self!r} is already closed')
        super().set_as_default(step)

    def as_default(self, step=None):
        """See `SummaryWriter.as_default`."""
        if context.executing_eagerly() and self._closed:
            raise RuntimeError(f'SummaryWriter {self!r} is already closed')
        return super().as_default(step)

    def init(self):
        """See `SummaryWriter.init`."""
        if context.executing_eagerly() and self._closed:
            raise RuntimeError(f'SummaryWriter {self!r} is already closed')
        return self._init_op

    def flush(self):
        """See `SummaryWriter.flush`."""
        if context.executing_eagerly() and self._closed:
            return
        with ops.device('cpu:0'):
            return gen_summary_ops.flush_summary_writer(self._resource)

    def close(self):
        """See `SummaryWriter.close`."""
        if context.executing_eagerly() and self._closed:
            return
        try:
            with ops.control_dependencies([self.flush()]):
                with ops.device('cpu:0'):
                    return gen_summary_ops.close_summary_writer(self._resource)
        finally:
            if context.executing_eagerly():
                self._closed = True