from pyudev.device import Device
class QUDevMonitorObserverMixin(MonitorObserverMixin):
    """
    Obsolete monitor observer mixin.
    """

    def _setup_notifier(self, monitor, notifier_class):
        MonitorObserverMixin._setup_notifier(self, monitor, notifier_class)
        self._action_signal_map = {'add': self.deviceAdded, 'remove': self.deviceRemoved, 'change': self.deviceChanged, 'move': self.deviceMoved}
        import warnings
        warnings.warn('Will be removed in 1.0. Use pyudev.pyqt4.MonitorObserver instead.', DeprecationWarning)

    def _emit_event(self, device):
        self.deviceEvent.emit(device.action, device)
        signal = self._action_signal_map.get(device.action)
        if signal is not None:
            signal.emit(device)