import sys
import ctypes
from pyglet.util import debug_print
class _pInterfaceMeta(_PointerMeta):

    def __new__(cls, name, bases, dct):
        target = dct.get('_type_', None)
        if target is None:
            interface_base = bases[0]._type_
            target = _InterfaceMeta(f'_{name}_HelperInterface', (interface_base,), {'_methods_': dct.get('_methods_', ())}, create_pointer_type=False)
            dct['_type_'] = target
        for i, (method_name, method) in enumerate(target._methods_):
            m = method.get_com_proxy(i + target.vtbl_own_offset, method_name)

            def pinterface_method_forward(self, *args, _m=m, _i=i):
                assert _debug_com(f'Calling COM {_i} of {target.__name__} ({_m}) through pointer: ({', '.join(map(repr, (self, *args)))})')
                return _m(self, *args)
            dct[method_name] = pinterface_method_forward
        pointer_type = super().__new__(cls, name, bases, dct)
        from ctypes import _pointer_type_cache
        _pointer_type_cache[target] = pointer_type
        return pointer_type