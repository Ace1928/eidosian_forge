import ctypes, ctypes.util, operator, sys
from . import model
class CTypesData(object):
    __metaclass__ = CTypesType
    __slots__ = ['__weakref__']
    __name__ = '<cdata>'

    def __init__(self, *args):
        raise TypeError('cannot instantiate %r' % (self.__class__,))

    @classmethod
    def _newp(cls, init):
        raise TypeError("expected a pointer or array ctype, got '%s'" % (cls._get_c_name(),))

    @staticmethod
    def _to_ctypes(value):
        raise TypeError

    @classmethod
    def _arg_to_ctypes(cls, *value):
        try:
            ctype = cls._ctype
        except AttributeError:
            raise TypeError('cannot create an instance of %r' % (cls,))
        if value:
            res = cls._to_ctypes(*value)
            if not isinstance(res, ctype):
                res = cls._ctype(res)
        else:
            res = cls._ctype()
        return res

    @classmethod
    def _create_ctype_obj(cls, init):
        if init is None:
            return cls._arg_to_ctypes()
        else:
            return cls._arg_to_ctypes(init)

    @staticmethod
    def _from_ctypes(ctypes_value):
        raise TypeError

    @classmethod
    def _get_c_name(cls, replace_with=''):
        return cls._reftypename.replace(' &', replace_with)

    @classmethod
    def _fix_class(cls):
        cls.__name__ = 'CData<%s>' % (cls._get_c_name(),)
        cls.__qualname__ = 'CData<%s>' % (cls._get_c_name(),)
        cls.__module__ = 'ffi'

    def _get_own_repr(self):
        raise NotImplementedError

    def _addr_repr(self, address):
        if address == 0:
            return 'NULL'
        else:
            if address < 0:
                address += 1 << 8 * ctypes.sizeof(ctypes.c_void_p)
            return '0x%x' % address

    def __repr__(self, c_name=None):
        own = self._get_own_repr()
        return '<cdata %r %s>' % (c_name or self._get_c_name(), own)

    def _convert_to_address(self, BClass):
        if BClass is None:
            raise TypeError('cannot convert %r to an address' % (self._get_c_name(),))
        else:
            raise TypeError('cannot convert %r to %r' % (self._get_c_name(), BClass._get_c_name()))

    @classmethod
    def _get_size(cls):
        return ctypes.sizeof(cls._ctype)

    def _get_size_of_instance(self):
        return ctypes.sizeof(self._ctype)

    @classmethod
    def _cast_from(cls, source):
        raise TypeError('cannot cast to %r' % (cls._get_c_name(),))

    def _cast_to_integer(self):
        return self._convert_to_address(None)

    @classmethod
    def _alignment(cls):
        return ctypes.alignment(cls._ctype)

    def __iter__(self):
        raise TypeError('cdata %r does not support iteration' % self._get_c_name())

    def _make_cmp(name):
        cmpfunc = getattr(operator, name)

        def cmp(self, other):
            v_is_ptr = not isinstance(self, CTypesGenericPrimitive)
            w_is_ptr = isinstance(other, CTypesData) and (not isinstance(other, CTypesGenericPrimitive))
            if v_is_ptr and w_is_ptr:
                return cmpfunc(self._convert_to_address(None), other._convert_to_address(None))
            elif v_is_ptr or w_is_ptr:
                return NotImplemented
            else:
                if isinstance(self, CTypesGenericPrimitive):
                    self = self._value
                if isinstance(other, CTypesGenericPrimitive):
                    other = other._value
                return cmpfunc(self, other)
        cmp.func_name = name
        return cmp
    __eq__ = _make_cmp('__eq__')
    __ne__ = _make_cmp('__ne__')
    __lt__ = _make_cmp('__lt__')
    __le__ = _make_cmp('__le__')
    __gt__ = _make_cmp('__gt__')
    __ge__ = _make_cmp('__ge__')

    def __hash__(self):
        return hash(self._convert_to_address(None))

    def _to_string(self, maxlen):
        raise TypeError('string(): %r' % (self,))