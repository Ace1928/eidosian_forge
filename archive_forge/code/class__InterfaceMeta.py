import sys
import ctypes
from pyglet.util import debug_print
class _InterfaceMeta(_StructMeta):

    def __new__(cls, name, bases, dct, /, create_pointer_type=True):
        if len(bases) > 1:
            assert _debug_com(f'Ignoring {len(bases) - 1} bases on {name}')
            bases = (bases[0],)
        if not '_methods_' in dct:
            dct['_methods_'] = ()
        inh_methods = []
        if bases[0] is not ctypes.Structure:
            for interface_type in bases[0].get_interface_inheritance():
                inh_methods.extend(interface_type.__dict__['_methods_'])
        inh_methods = tuple(inh_methods)
        new_methods = tuple(dct['_methods_'])
        vtbl_own_offset = len(inh_methods)
        all_methods = tuple(inh_methods) + new_methods
        for i, (method_name, mt) in enumerate(all_methods):
            assert _debug_com(f'{name}[{i}]: {method_name}: {', '.join((t.__name__ for t in mt.argtypes)) or 'void'} -> {('void' if mt.restype is None else mt.restype.__name__)}')
        vtbl_struct_type = _StructMeta(f'Vtable_{name}', (ctypes.Structure,), {'_fields_': [(n, x.direct_prototype) for n, x in all_methods]})
        dct['_vtbl_struct_type'] = vtbl_struct_type
        dct['vtbl_own_offset'] = vtbl_own_offset
        dct['_fields_'] = (('vtbl_ptr', ctypes.POINTER(vtbl_struct_type)),)
        res_type = super().__new__(cls, name, bases, dct)
        if create_pointer_type:
            _pInterfaceMeta(f'p{name}', (ctypes.POINTER(bases[0]),), {'_type_': res_type})
        return res_type