import abc
class DeviceNotFoundByInterfaceIndexError(DeviceNotFoundError):
    """
    A :exc:`DeviceNotFoundError` indicating that no :class:`Device` was found
    from the given interface index.
    """