from wx import EvtHandler, PostEvent  # pylint: disable=import-error
from wx.lib.newevent import NewEvent  # pylint: disable=import-error, no-name-in-module
import pyudev  # pylint: disable=wrong-import-order
class WxUDevMonitorObserver(MonitorObserver):
    """An observer for device events integrating into the :mod:`wx` mainloop.

    .. deprecated:: 0.17
       Will be removed in 1.0.  Use :class:`MonitorObserver` instead.
    """
    _action_event_map = {'add': DeviceAddedEvent, 'remove': DeviceRemovedEvent, 'change': DeviceChangedEvent, 'move': DeviceMovedEvent}

    def __init__(self, monitor):
        MonitorObserver.__init__(self, monitor)
        import warnings
        warnings.warn('Will be removed in 1.0. Use pyudev.wx.MonitorObserver instead.', DeprecationWarning)

    def _emit_event(self, device):
        PostEvent(self, DeviceEvent(action=device.action, device=device))
        event_class = self._action_event_map.get(device.action)
        if event_class is not None:
            PostEvent(self, event_class(device=device))