from gi.repository import GLib, GObject  # pylint: disable=import-error
class GUDevMonitorObserver(GObject.Object, _ObserverMixin):
    """
    An observer for device events integrating into the :mod:`gi.repository.GLib`
    mainloop.

    .. deprecated:: 0.17
       Will be removed in 1.0.  Use :class:`MonitorObserver` instead.
    """
    _action_signal_map = {'add': 'device-added', 'remove': 'device-removed', 'change': 'device-changed', 'move': 'device-moved'}
    __gsignals__ = {str('device-event'): (GObject.SIGNAL_RUN_LAST, GObject.TYPE_NONE, (GObject.TYPE_STRING, GObject.TYPE_PYOBJECT)), str('device-added'): (GObject.SIGNAL_RUN_LAST, GObject.TYPE_NONE, (GObject.TYPE_PYOBJECT,)), str('device-removed'): (GObject.SIGNAL_RUN_LAST, GObject.TYPE_NONE, (GObject.TYPE_PYOBJECT,)), str('device-changed'): (GObject.SIGNAL_RUN_LAST, GObject.TYPE_NONE, (GObject.TYPE_PYOBJECT,)), str('device-moved'): (GObject.SIGNAL_RUN_LAST, GObject.TYPE_NONE, (GObject.TYPE_PYOBJECT,))}

    def __init__(self, monitor):
        GObject.Object.__init__(self)
        self._setup_observer(monitor)
        import warnings
        warnings.warn('Will be removed in 1.0. Use pyudev.glib.MonitorObserver instead.', DeprecationWarning)

    def _emit_event(self, device):
        self.emit('device-event', device.action, device)
        signal = self._action_signal_map.get(device.action)
        if signal is not None:
            self.emit(signal, device)