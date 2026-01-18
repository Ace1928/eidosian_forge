import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
def _gdk_atom_repr(atom):
    n = atom.name()
    if n:
        return 'Gdk.Atom.intern("%s", False)' % n
    return '<Gdk.Atom(%i)>' % hash(atom)