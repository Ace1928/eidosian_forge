import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
class _BaseDisplay(display.Display):
    resource_classes = _resource_baseclasses.copy()

    def __init__(self, *args, **keys):
        display.Display.__init__(*(self,) + args, **keys)
        self._atom_cache = {}

    def get_atom(self, atomname, only_if_exists=0):
        if atomname in self._atom_cache:
            return self._atom_cache[atomname]
        r = request.InternAtom(display=self, name=atomname, only_if_exists=only_if_exists)
        if r.atom != X.NONE:
            self._atom_cache[atomname] = r.atom
        return r.atom