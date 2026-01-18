import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
class FileEnumerator(Gio.FileEnumerator):

    def __iter__(self):
        return self

    def __next__(self):
        file_info = self.next_file(None)
        if file_info is not None:
            return file_info
        else:
            raise StopIteration
    next = __next__