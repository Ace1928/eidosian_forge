import traceback
import warnings
import collections
from array import array
from threading import Lock
import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, c_int, c_char, c_char_p, cast, c_bool
from .base import _LIB, check_call, MXCallbackList, c_array, c_array_buf, mx_int, OpHandle
from .base import c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
from . import symbol, context
from .ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ID_TO_STR
from .ndarray.ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray.ndarray import _STORAGE_TYPE_CSR, _STORAGE_TYPE_ROW_SPARSE
from .ndarray import _ndarray_cls
from .numpy.multiarray import _np_ndarray_cls
from .util import is_np_array
class CustomOpProp(object):
    """Base class for operator property class implemented in python.

    Parameters
    ----------
    need_top_grad : bool
        The default declare_backward_dependency function. Use this value
        to determine whether this operator needs gradient input.
    """

    def __init__(self, need_top_grad=True):
        self.need_top_grad_ = need_top_grad

    def infer_shape(self, in_shape):
        """infer_shape interface. Can override when creating new operators.

        Parameters
        ----------
        in_shape : list
            List of argument shapes in the same order as
            declared in list_arguments.

        Returns
        -------
        in_shape : list
            List of argument shapes. Can be modified from in_shape.
        out_shape : list
            List of output shapes calculated from in_shape,
            in the same order as declared in list_outputs.
        aux_shape : Optional, list
            List of aux shapes calculated from in_shape,
            in the same order as declared in list_auxiliary_states.
        """
        return (in_shape, (in_shape[0],) * len(self.list_outputs()), ())

    def infer_type(self, in_type):
        """infer_type interface. override to create new operators

        Parameters
        ----------
        in_type : list of np.dtype
            list of argument types in the same order as
            declared in list_arguments.

        Returns
        -------
        in_type : list
            list of argument types. Can be modified from in_type.
        out_type : list
            list of output types calculated from in_type,
            in the same order as declared in list_outputs.
        aux_type : Optional, list
            list of aux types calculated from in_type,
            in the same order as declared in list_auxiliary_states.
        """
        return (in_type, [in_type[0]] * len(self.list_outputs()), [in_type[0]] * len(self.list_auxiliary_states()))

    def infer_storage_type(self, in_stype):
        """infer_storage_type interface. Used to infer storage type of
        inputs and outputs in the forward pass. When this interface is not implemented,
        all stypes will be inferred as default.

        Parameters
        ----------
        in_stype : list of stypes, valid stypes are default, row_sparse and
            csr

        Returns
        -------
        in_stype : list
            list of argument stypes.
        out_stype : list
            list of output types calculated from in_stype,
            in the same order as declared in list_outputs.
        aux_type : Optional, list
            list of aux types calculated from in_stype,
            in the same order as declared in list_auxiliary_states.
        """
        for i, stype in enumerate(in_stype):
            assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], "Default infer_storage_type implementation doesnt allow non default stypes: found non default stype '%s' for in_stype[%d]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default input/output stypes" % (stype, i)
        return (in_stype, [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]] * len(self.list_outputs()), [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]] * len(self.list_auxiliary_states()))

    def infer_storage_type_backward(self, ograd_stype, in_stype, out_stype, igrad_stype, aux_stype):
        """infer_storage_type_backward interface. Used to infer storage
        type of inputs and outputs in the backward pass.

        Will raise an error if undefined storage type is returned.
        Returned lists have to be the same size as the input lists to infer_storage_type_backward,
        otherwise an exception will be thrown. When this interface is not implemented,
        all stypes will be inferred as default.

        Parameters
        ----------
        ograd_stype : list
            list of output gradient storage types
        in_stype : list
            list of input storage types
        out_stype : list
            list of output storage types
        igrad_stype : list
            list of input gradient storage types
        aux_stype : list
            list of auxiliary storage types

        Returns
        -------
        ograd_stype : list
            list of inferred output gradient storage types
        in_stype : list
            list of inferred input storage types
        out_stype : list
            list of inferred output storage types
        igrad_stype : list
            list of inferred input gradient storage types
        aux_stype : list
            list of inferred storage types for auxiliary states
        """
        for i, stype in enumerate(ograd_stype):
            assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], "Default infer_storage_type_backward implementation doesnt allow non default stypes: found non default stype '%s' for ograd_stype[%d]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default output gradient stypes" % (stype, i)
        for i, stype in enumerate(igrad_stype):
            if stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_UNDEFINED]:
                stype = _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]
            assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], "Default infer_storage_type_backward implementation doesnt allow non default stypes: found non default stype '%s' for igrad_stype[%d]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default input gradient stypes" % (stype, i)
        stype_lists = [ograd_stype, in_stype, out_stype, igrad_stype, aux_stype]
        for stype_list in stype_lists:
            stype_list[:] = len(stype_list) * [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]]
        return (stype_lists[0], stype_lists[1], stype_lists[2], stype_lists[3], stype_lists[4])

    def list_outputs(self):
        """list_outputs interface. Can override when creating new operators.

        Returns
        -------
        outputs : list
            List of output blob names.
        """
        return ['output']

    def list_arguments(self):
        """list_arguments interface. Can override when creating new operators.

        Returns
        -------
        arguments : list
            List of argument blob names.
        """
        return ['data']

    def list_auxiliary_states(self):
        """list_auxiliary_states interface. Can override when creating new operators.

        Returns
        -------
        auxs : list
            list of auxiliary state blob names.
        """
        return []

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        """Declare dependencies of this operator for backward pass.

        Parameters
        ----------
        out_grad : list of int
            ids of out_grad blobs.
        in_data : list of int
            ids of in_data blobs.
        out_data: list of int
            ids of out_data blobs.

        Returns
        -------
        deps : list of int
            ids of the needed blobs.
        """
        deps = []
        if self.need_top_grad_:
            deps.extend(out_grad)
        deps.extend(in_data)
        deps.extend(out_data)
        return deps

    def create_operator(self, ctx, in_shapes, in_dtypes):
        """Create an operator that carries out the real computation
        given the context, input shapes, and input data types."""
        return CustomOp()