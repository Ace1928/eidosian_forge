import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
class _RecursiveAttr(object):
    """
    RecursiveAttr class to catch all attributes corresponding to numpy,
    when user calls fallback_mode. numpy is an instance of this class.
    """

    def __init__(self, numpy_object, cupy_object, array=None):
        """
        _RecursiveAttr initializer.

        Args:
            numpy_object (method): NumPy method.
            cupy_method (method): Corresponding CuPy method.
            array (ndarray): Acts as flag to know if _RecursiveAttr object
                is called from ``ndarray`` class. Also, acts as container for
                modifying args in case it is called from ``ndarray``.
                None otherwise.
        """
        self._numpy_object = numpy_object
        self._cupy_object = cupy_object
        self._fallback_array = array

    def __instancecheck__(self, instance):
        """
        Enable support for isinstance(instance, _RecursiveAttr instance)
        by redirecting it to appropriate isinstance method.
        """
        if self._cupy_object is not None:
            return isinstance(instance, self._cupy_object)
        return isinstance(instance, self._numpy_object)

    def __getattr__(self, attr):
        """
        Catches attributes corresponding to numpy.

        Runs recursively till attribute gets called.
        Or numpy ScalarType is retrieved.

        Args:
            attr (str): Attribute of _RecursiveAttr class object.

        Returns:
            (_RecursiveAttr object, NumPy scalar):
                Returns_RecursiveAttr object with new numpy_object,
                cupy_object. OR
                Returns objects in cupy which is an alias
                of numpy object. OR
                Returns wrapper objects, `ndarray`, `vectorize`.
        """
        numpy_object = getattr(self._numpy_object, attr)
        cupy_object = getattr(self._cupy_object, attr, None)
        if numpy_object is np.ndarray:
            return ndarray
        if numpy_object is np.vectorize:
            return vectorize
        if numpy_object is cupy_object:
            return numpy_object
        return _RecursiveAttr(numpy_object, cupy_object)

    def __repr__(self):
        if isinstance(self._numpy_object, types.ModuleType):
            return '<numpy = module {}, cupy = module {}>'.format(self._numpy_object.__name__, getattr(self._cupy_object, '__name__', None))
        return '<numpy = {}, cupy = {}>'.format(self._numpy_object, self._cupy_object)

    @property
    def __doc__(self):
        return self._numpy_object.__doc__

    @staticmethod
    def _is_cupy_compatible(arg):
        """
        Returns False if CuPy's functions never accept the arguments as
        parameters due to the following reasons.
        - The inputs include an object of a NumPy's specific class other than
          `np.ndarray`.
        - The inputs include a dtype which is not supported in CuPy.
        """
        if isinstance(arg, ndarray):
            if not arg._supports_cupy:
                return False
        if isinstance(arg, (tuple, list)):
            return all((_RecursiveAttr._is_cupy_compatible(i) for i in arg))
        if isinstance(arg, dict):
            bools = [_RecursiveAttr._is_cupy_compatible(arg[i]) for i in arg]
            return all(bools)
        return True

    def __call__(self, *args, **kwargs):
        """
        Gets invoked when last attribute of _RecursiveAttr class gets called.
        Calls _cupy_object if not None else call _numpy_object.

        Args:
            args (tuple): Arguments.
            kwargs (dict): Keyword arguments.

        Returns:
            (res, ndarray): Returns of methods call_cupy or call_numpy
        """
        if not callable(self._numpy_object):
            raise TypeError("'{}' object is not callable".format(type(self._numpy_object).__name__))
        if self._fallback_array is not None:
            args = (self._fallback_array,) + args
        if self._cupy_object is not None and _RecursiveAttr._is_cupy_compatible((args, kwargs)):
            try:
                return _call_cupy(self._cupy_object, args, kwargs)
            except Exception:
                return _call_numpy(self._numpy_object, args, kwargs)
        notification._dispatch_notification(self._numpy_object)
        return _call_numpy(self._numpy_object, args, kwargs)