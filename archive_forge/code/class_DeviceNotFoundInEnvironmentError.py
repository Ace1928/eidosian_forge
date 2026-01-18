import abc
class DeviceNotFoundInEnvironmentError(DeviceNotFoundError):
    """
    A :exc:`DeviceNotFoundError` indicating, that no :class:`Device` could
    be constructed from the process environment.
    """

    def __str__(self):
        return 'No device found in environment'