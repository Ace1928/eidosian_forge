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
class DataDesc(namedtuple('DataDesc', ['name', 'shape'])):
    """DataDesc is used to store name, shape, type and layout
    information of the data or the label.

    The `layout` describes how the axes in `shape` should be interpreted,
    for example for image data setting `layout=NCHW` indicates
    that the first axis is number of examples in the batch(N),
    C is number of channels, H is the height and W is the width of the image.

    For sequential data, by default `layout` is set to ``NTC``, where
    N is number of examples in the batch, T the temporal axis representing time
    and C is the number of channels.

    Parameters
    ----------
    cls : DataDesc
         The class.
    name : str
         Data name.
    shape : tuple of int
         Data shape.
    dtype : np.dtype, optional
         Data type.
    layout : str, optional
         Data layout.
    """

    def __new__(cls, name, shape, dtype=mx_real_t, layout='NCHW'):
        ret = super(cls, DataDesc).__new__(cls, name, shape)
        ret.dtype = dtype
        ret.layout = layout
        return ret

    def __repr__(self):
        return 'DataDesc[%s,%s,%s,%s]' % (self.name, self.shape, self.dtype, self.layout)

    @staticmethod
    def get_batch_axis(layout):
        """Get the dimension that corresponds to the batch size.

        When data parallelism is used, the data will be automatically split and
        concatenated along the batch-size dimension. Axis can be -1, which means
        the whole array will be copied for each data-parallelism device.

        Parameters
        ----------
        layout : str
            layout string. For example, "NCHW".

        Returns
        -------
        int
            An axis indicating the batch_size dimension.
        """
        if layout is None:
            return 0
        return layout.find('N')

    @staticmethod
    def get_list(shapes, types):
        """Get DataDesc list from attribute lists.

        Parameters
        ----------
        shapes : a tuple of (name_, shape_)
        types : a tuple of  (name_, np.dtype)
        """
        if types is not None:
            type_dict = dict(types)
            return [DataDesc(x[0], x[1], type_dict[x[0]]) for x in shapes]
        else:
            return [DataDesc(x[0], x[1]) for x in shapes]