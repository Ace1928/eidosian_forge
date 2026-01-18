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
class NDArrayOp(PythonOp):
    """Base class for numpy operators. numpy operators allow parts
    of computation in symbolic graph to be writen in numpy. This feature
    is intended for quickly hacking out a solution for non performance
    critical parts. Please consider write a c++ implementation if it becomes
    a bottleneck.
    Note that if your operator contains internal states (like arrays),
    it cannot be used for multi-gpu training.
    """

    def __init__(self, need_top_grad=True):
        super(NDArrayOp, self).__init__(need_top_grad)
        warnings.warn('NDArrayOp has been deprecated. Please use CustomOp')

    def get_symbol(self, *args, **kwargs):
        fb_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_void_p), POINTER(c_int), c_void_p)
        infer_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_int), POINTER(POINTER(mx_int)), c_void_p)
        list_functype = CFUNCTYPE(c_bool, POINTER(POINTER(POINTER(c_char))), c_void_p)
        deps_functype = CFUNCTYPE(c_bool, c_int_p, c_int_p, c_int_p, c_int_p, POINTER(c_int_p), c_void_p)

        class NDArrayOpInfo(Structure):
            """Structure that holds Callback information. Passed to NDArrayOpProp"""
            _fields_ = [('forward', fb_functype), ('backward', fb_functype), ('infer_shape', infer_functype), ('list_outputs', list_functype), ('list_arguments', list_functype), ('declare_backward_dependency', deps_functype), ('p_forward', c_void_p), ('p_backward', c_void_p), ('p_infer_shape', c_void_p), ('p_list_outputs', c_void_p), ('p_list_arguments', c_void_p), ('p_declare_backward_dependency', c_void_p)]

        def forward_entry(num_ndarray, ndarraies, tags, _):
            """C Callback for NDArrayOp::Forward"""
            try:
                tensors = [[] for i in range(4)]
                for i in range(num_ndarray):
                    if tags[i] == 1:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle), writable=True))
                    else:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle), writable=False))
                self.forward(in_data=tensors[0], out_data=tensors[1])
            except Exception:
                print('Error in NDArrayOp.forward: %s' % traceback.format_exc())
                return False
            return True

        def backward_entry(num_ndarray, ndarraies, tags, _):
            """C Callback for NDArrayOp::Backward"""
            try:
                tensors = [[] for i in range(4)]
                for i in range(num_ndarray):
                    if tags[i] == 2:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle), writable=True))
                    else:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle), writable=False))
                self.backward(in_data=tensors[0], out_data=tensors[1], in_grad=tensors[2], out_grad=tensors[3])
            except Exception:
                print('Error in NDArrayOp.backward: %s' % traceback.format_exc())
                return False
            return True

        def infer_shape_entry(num_tensor, tensor_dims, tensor_shapes, _):
            """C Callback for NDArrayOpProp::InferShape"""
            try:
                n_in = len(self.list_arguments())
                n_out = len(self.list_outputs())
                assert num_tensor == n_in + n_out
                shapes = [[tensor_shapes[i][j] for j in range(tensor_dims[i])] for i in range(n_in)]
                ishape, oshape = self.infer_shape(shapes)
                assert len(oshape) == n_out
                assert len(ishape) == n_in
                rshape = list(ishape) + list(oshape)
                for i in range(n_in + n_out):
                    tensor_shapes[i] = cast(c_array_buf(mx_int, array('i', rshape[i])), POINTER(mx_int))
                    tensor_dims[i] = len(rshape[i])
            except Exception:
                print('Error in NDArrayOp.infer_shape: %s' % traceback.format_exc())
                return False
            return True

        def list_outputs_entry(out, _):
            """C Callback for NDArrayOpProp::ListOutputs"""
            try:
                ret = self.list_outputs()
                ret = [c_str(i) for i in ret] + [c_char_p(0)]
                ret = c_array(c_char_p, ret)
                out[0] = cast(ret, POINTER(POINTER(c_char)))
            except Exception:
                print('Error in NDArrayOp.list_outputs: %s' % traceback.format_exc())
                return False
            return True

        def list_arguments_entry(out, _):
            """C Callback for NDArrayOpProp::ListArguments"""
            try:
                ret = self.list_arguments()
                ret = [c_str(i) for i in ret] + [c_char_p(0)]
                ret = c_array(c_char_p, ret)
                out[0] = cast(ret, POINTER(POINTER(c_char)))
            except Exception:
                print('Error in NDArrayOp.list_arguments: %s' % traceback.format_exc())
                return False
            return True

        def declare_backward_dependency(out_grad, in_data, out_data, num_dep, deps, _):
            """C Callback for NDArrayOpProp::DeclareBacwardDependency"""
            try:
                out_grad = [out_grad[i] for i in range(len(self.list_outputs()))]
                in_data = [in_data[i] for i in range(len(self.list_arguments()))]
                out_data = [out_data[i] for i in range(len(self.list_outputs()))]
                rdeps = self.declare_backward_dependency(out_grad, in_data, out_data)
                num_dep[0] = len(rdeps)
                rdeps = cast(c_array_buf(c_int, array('i', rdeps)), c_int_p)
                deps[0] = rdeps
            except Exception:
                print('Error in NDArrayOp.declare_backward_dependency: %s' % traceback.format_exc())
                return False
            return True
        self.info_ = NDArrayOpInfo(fb_functype(forward_entry), fb_functype(backward_entry), infer_functype(infer_shape_entry), list_functype(list_outputs_entry), list_functype(list_arguments_entry), deps_functype(declare_backward_dependency), None, None, None, None, None, None)
        cb_ptr = format(cast(pointer(self.info_), c_void_p).value, 'x')
        sym = symbol._internal._NDArray(*args, info=cb_ptr, **kwargs)
        PythonOp._ref_holder.append(self)
        return sym

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
        if self.need_top_grad():
            deps.extend(out_grad)
        deps.extend(in_data)
        deps.extend(out_data)
        return deps