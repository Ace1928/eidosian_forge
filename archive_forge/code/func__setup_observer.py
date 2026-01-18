from gi.repository import GLib, GObject  # pylint: disable=import-error
def _setup_observer(self, monitor):
    self.monitor = monitor
    self.event_source = None
    self.enabled = True