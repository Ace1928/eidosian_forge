from collections import namedtuple
import sys
import ctypes
import logging
import threading
import numpy as np
from ..base import _LIB
from ..base import c_str_array, mx_uint, py_str
from ..base import DataIterHandle, NDArrayHandle
from ..base import mx_real_t
from ..base import check_call, build_param_doc as _build_param_doc
from ..ndarray import NDArray
from ..ndarray.sparse import CSRNDArray
from ..ndarray import _ndarray_cls
from ..ndarray import array
from ..ndarray import concat, tile
from .utils import _init_data, _has_instance, _getdata_by_idx, _slice_along_batch_axis
class DataBatch(object):
    """A data batch.

    MXNet's data iterator returns a batch of data for each `next` call.
    This data contains `batch_size` number of examples.

    If the input data consists of images, then shape of these images depend on
    the `layout` attribute of `DataDesc` object in `provide_data` parameter.

    If `layout` is set to 'NCHW' then, images should be stored in a 4-D matrix
    of shape ``(batch_size, num_channel, height, width)``.
    If `layout` is set to 'NHWC' then, images should be stored in a 4-D matrix
    of shape ``(batch_size, height, width, num_channel)``.
    The channels are often in RGB order.

    Parameters
    ----------
    data : list of `NDArray`, each array containing `batch_size` examples.
          A list of input data.
    label : list of `NDArray`, each array often containing a 1-dimensional array. optional
          A list of input labels.
    pad : int, optional
          The number of examples padded at the end of a batch. It is used when the
          total number of examples read is not divisible by the `batch_size`.
          These extra padded examples are ignored in prediction.
    index : numpy.array, optional
          The example indices in this batch.
    bucket_key : int, optional
          The bucket key, used for bucketing module.
    provide_data : list of `DataDesc`, optional
          A list of `DataDesc` objects. `DataDesc` is used to store
          name, shape, type and layout information of the data.
          The *i*-th element describes the name and shape of ``data[i]``.
    provide_label : list of `DataDesc`, optional
          A list of `DataDesc` objects. `DataDesc` is used to store
          name, shape, type and layout information of the label.
          The *i*-th element describes the name and shape of ``label[i]``.
    """

    def __init__(self, data, label=None, pad=None, index=None, bucket_key=None, provide_data=None, provide_label=None):
        if data is not None:
            assert isinstance(data, (list, tuple)), 'Data must be list of NDArrays'
        if label is not None:
            assert isinstance(label, (list, tuple)), 'Label must be list of NDArrays'
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index
        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label

    def __str__(self):
        data_shapes = [d.shape for d in self.data]
        if self.label:
            label_shapes = [l.shape for l in self.label]
        else:
            label_shapes = None
        return '{}: data shapes: {} label shapes: {}'.format(self.__class__.__name__, data_shapes, label_shapes)