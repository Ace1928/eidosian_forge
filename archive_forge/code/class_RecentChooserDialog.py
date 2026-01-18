import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class RecentChooserDialog(Gtk.RecentChooserDialog):
    __init__ = deprecated_init(Gtk.RecentChooserDialog.__init__, arg_names=('title', 'parent', 'recent_manager', 'buttons'), deprecated_aliases={'recent_manager': 'manager'}, category=PyGTKDeprecationWarning)