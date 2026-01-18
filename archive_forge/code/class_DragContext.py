import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
class DragContext(Gdk.DragContext):

    def finish(self, success, del_, time):
        Gtk = get_introspection_module('Gtk')
        Gtk.drag_finish(self, success, del_, time)