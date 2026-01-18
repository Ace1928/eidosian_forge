from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def _get_struct_prop(self, pname, ptype, pstruct):
    r = self.get_property(pname, ptype, 0, pstruct.static_size // 4)
    if r and r.format == 32:
        value = r.value.tostring()
        if len(value) == pstruct.static_size:
            return pstruct.parse_binary(value, self.display)[0]
    return None