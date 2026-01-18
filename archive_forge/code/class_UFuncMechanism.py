from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
class UFuncMechanism(object):
    """
    Prepare ufunc arguments for vectorize.
    """
    DEFAULT_STREAM = None
    SUPPORT_DEVICE_SLICING = False

    def __init__(self, typemap, args):
        """Never used directly by user. Invoke by UFuncMechanism.call().
        """
        self.typemap = typemap
        self.args = args
        nargs = len(self.args)
        self.argtypes = [None] * nargs
        self.scalarpos = []
        self.signature = None
        self.arrays = [None] * nargs

    def _fill_arrays(self):
        """
        Get all arguments in array form
        """
        for i, arg in enumerate(self.args):
            if self.is_device_array(arg):
                self.arrays[i] = self.as_device_array(arg)
            elif isinstance(arg, (int, float, complex, np.number)):
                self.scalarpos.append(i)
            else:
                self.arrays[i] = np.asarray(arg)

    def _fill_argtypes(self):
        """
        Get dtypes
        """
        for i, ary in enumerate(self.arrays):
            if ary is not None:
                dtype = getattr(ary, 'dtype')
                if dtype is None:
                    dtype = np.asarray(ary).dtype
                self.argtypes[i] = dtype

    def _resolve_signature(self):
        """Resolve signature.
        May have ambiguous case.
        """
        matches = []
        if self.scalarpos:
            for formaltys in self.typemap:
                match_map = []
                for i, (formal, actual) in enumerate(zip(formaltys, self.argtypes)):
                    if actual is None:
                        actual = np.asarray(self.args[i]).dtype
                    match_map.append(actual == formal)
                if all(match_map):
                    matches.append(formaltys)
        if not matches:
            matches = []
            for formaltys in self.typemap:
                all_matches = all((actual is None or formal == actual for formal, actual in zip(formaltys, self.argtypes)))
                if all_matches:
                    matches.append(formaltys)
        if not matches:
            raise TypeError("No matching version.  GPU ufunc requires array arguments to have the exact types.  This behaves like regular ufunc with casting='no'.")
        if len(matches) > 1:
            raise TypeError('Failed to resolve ufunc due to ambiguous signature. Too many untyped scalars. Use numpy dtype object to type tag.')
        self.argtypes = matches[0]

    def _get_actual_args(self):
        """Return the actual arguments
        Casts scalar arguments to np.array.
        """
        for i in self.scalarpos:
            self.arrays[i] = np.array([self.args[i]], dtype=self.argtypes[i])
        return self.arrays

    def _broadcast(self, arys):
        """Perform numpy ufunc broadcasting
        """
        shapelist = [a.shape for a in arys]
        shape = _multi_broadcast(*shapelist)
        for i, ary in enumerate(arys):
            if ary.shape == shape:
                pass
            elif self.is_device_array(ary):
                arys[i] = self.broadcast_device(ary, shape)
            else:
                ax_differs = [ax for ax in range(len(shape)) if ax >= ary.ndim or ary.shape[ax] != shape[ax]]
                missingdim = len(shape) - len(ary.shape)
                strides = [0] * missingdim + list(ary.strides)
                for ax in ax_differs:
                    strides[ax] = 0
                strided = np.lib.stride_tricks.as_strided(ary, shape=shape, strides=strides)
                arys[i] = self.force_array_layout(strided)
        return arys

    def get_arguments(self):
        """Prepare and return the arguments for the ufunc.
        Does not call to_device().
        """
        self._fill_arrays()
        self._fill_argtypes()
        self._resolve_signature()
        arys = self._get_actual_args()
        return self._broadcast(arys)

    def get_function(self):
        """Returns (result_dtype, function)
        """
        return self.typemap[self.argtypes]

    def is_device_array(self, obj):
        """Is the `obj` a device array?
        Override in subclass
        """
        return False

    def as_device_array(self, obj):
        """Convert the `obj` to a device array
        Override in subclass

        Default implementation is an identity function
        """
        return obj

    def broadcast_device(self, ary, shape):
        """Handles ondevice broadcasting

        Override in subclass to add support.
        """
        raise NotImplementedError('broadcasting on device is not supported')

    def force_array_layout(self, ary):
        """Ensures array layout met device requirement.

        Override in sublcass
        """
        return ary

    @classmethod
    def call(cls, typemap, args, kws):
        """Perform the entire ufunc call mechanism.
        """
        stream = kws.pop('stream', cls.DEFAULT_STREAM)
        out = kws.pop('out', None)
        if kws:
            warnings.warn('unrecognized keywords: %s' % ', '.join(kws))
        cr = cls(typemap, args)
        args = cr.get_arguments()
        resty, func = cr.get_function()
        outshape = args[0].shape
        if out is not None and cr.is_device_array(out):
            out = cr.as_device_array(out)

        def attempt_ravel(a):
            if cr.SUPPORT_DEVICE_SLICING:
                raise NotImplementedError
            try:
                return a.ravel()
            except NotImplementedError:
                if not cr.is_device_array(a):
                    raise
                else:
                    hostary = cr.to_host(a, stream).ravel()
                    return cr.to_device(hostary, stream)
        if args[0].ndim > 1:
            args = [attempt_ravel(a) for a in args]
        devarys = []
        any_device = False
        for a in args:
            if cr.is_device_array(a):
                devarys.append(a)
                any_device = True
            else:
                dev_a = cr.to_device(a, stream=stream)
                devarys.append(dev_a)
        shape = args[0].shape
        if out is None:
            devout = cr.allocate_device_array(shape, resty, stream=stream)
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            if any_device:
                return devout.reshape(outshape)
            else:
                return devout.copy_to_host().reshape(outshape)
        elif cr.is_device_array(out):
            if out.ndim > 1:
                out = attempt_ravel(out)
            devout = out
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.reshape(outshape)
        else:
            assert out.shape == shape
            assert out.dtype == resty
            devout = cr.allocate_device_array(shape, resty, stream=stream)
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.copy_to_host(out, stream=stream).reshape(outshape)

    def to_device(self, hostary, stream):
        """Implement to device transfer
        Override in subclass
        """
        raise NotImplementedError

    def to_host(self, devary, stream):
        """Implement to host transfer
        Override in subclass
        """
        raise NotImplementedError

    def allocate_device_array(self, shape, dtype, stream):
        """Implements device allocation
        Override in subclass
        """
        raise NotImplementedError

    def launch(self, func, count, stream, args):
        """Implements device function invocation
        Override in subclass
        """
        raise NotImplementedError