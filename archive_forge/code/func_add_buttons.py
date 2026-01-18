import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def add_buttons(self, *args):
    """
        The add_buttons() method adds several buttons to the Gtk.Dialog using
        the button data passed as arguments to the method. This method is the
        same as calling the Gtk.Dialog.add_button() repeatedly. The button data
        pairs - button text (or stock ID) and a response ID integer are passed
        individually. For example:

        .. code-block:: python

            dialog.add_buttons(Gtk.STOCK_OPEN, 42, "Close", Gtk.ResponseType.CLOSE)

        will add "Open" and "Close" buttons to dialog.
        """

    def _button(b):
        while b:
            try:
                t, r = b[0:2]
            except ValueError:
                raise ValueError('Must pass an even number of arguments')
            b = b[2:]
            yield (t, r)
    for text, response in _button(args):
        self.add_button(text, response)