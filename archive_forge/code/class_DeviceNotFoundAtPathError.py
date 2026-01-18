import abc
class DeviceNotFoundAtPathError(DeviceNotFoundError):
    """
    A :exc:`DeviceNotFoundError` indicating that no :class:`Device` was
    found at a given path.
    """

    def __init__(self, sys_path):
        DeviceNotFoundError.__init__(self, sys_path)

    @property
    def sys_path(self):
        """
        The path that caused this error as string.
        """
        return self.args[0]

    def __str__(self):
        return 'No device at {0!r}'.format(self.sys_path)