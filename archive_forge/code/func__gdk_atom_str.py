import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
def _gdk_atom_str(atom):
    n = atom.name()
    if n:
        return n
    return 'Gdk.Atom<%i>' % hash(atom)