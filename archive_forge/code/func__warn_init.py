import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
def _warn_init(cls, instead=None):

    def new_init(self, *args, **kwargs):
        super(cls, self).__init__(*args, **kwargs)
        name = cls.__module__.rsplit('.', 1)[-1] + '.' + cls.__name__
        if instead:
            warnings.warn("%s shouldn't be instantiated directly, use %s instead." % (name, instead), PyGIWarning, stacklevel=2)
        else:
            warnings.warn("%s shouldn't be instantiated directly." % (name,), PyGIWarning, stacklevel=2)
    return new_init