from . import _gi
class SignalOverride(Signal):
    """Specialized sub-class of Signal which can be used as a decorator for overriding
    existing signals on GObjects.

    :Example:

    .. code-block:: python

        class MyWidget(Gtk.Widget):
            @GObject.SignalOverride
            def configure_event(self):
                pass
    """

    def get_signal_args(self):
        """Returns the string 'override'."""
        return 'override'