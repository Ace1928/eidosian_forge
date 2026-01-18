import abc
class DeviceNotFoundByKernelDeviceError(DeviceNotFoundError):
    """
    A :exc:`DeviceNotFoundError` indicating that no :class:`Device` was found
    from the given kernel device string.

    The format of the kernel device string is defined in the
    systemd.journal-fields man pages.
    """