from pyudev.device import Device
Generates an observer for device events integrating into the
        PyQt{4,5} mainloop.

        This class inherits :class:`~PyQt{4,5}.QtCore.QObject` to turn device
        events into Qt signals:

        >>> from pyudev import Context, Monitor
        >>> from pyudev.pyqt4 import MonitorObserver
        >>> context = Context()
        >>> monitor = Monitor.from_netlink(context)
        >>> monitor.filter_by(subsystem='input')
        >>> observer = MonitorObserver(monitor)
        >>> def device_event(device):
        ...     print('event {0} on device {1}'.format(device.action, device))
        >>> observer.deviceEvent.connect(device_event)
        >>> monitor.start()

        This class is a child of :class:`~{PyQt{4,5}, PySide}.QtCore.QObject`.

        