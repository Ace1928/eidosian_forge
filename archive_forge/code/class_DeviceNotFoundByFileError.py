import abc
class DeviceNotFoundByFileError(DeviceNotFoundError):
    """
    A :exc:`DeviceNotFoundError` indicating that no :class:`Device` was
    found from the given filename.
    """