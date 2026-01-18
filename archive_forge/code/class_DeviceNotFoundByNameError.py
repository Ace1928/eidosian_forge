import abc
class DeviceNotFoundByNameError(DeviceNotFoundError):
    """
    A :exc:`DeviceNotFoundError` indicating that no :class:`Device` was
    found with a given name.
    """

    def __init__(self, subsystem, sys_name):
        DeviceNotFoundError.__init__(self, subsystem, sys_name)

    @property
    def subsystem(self):
        """
        The subsystem that caused this error as string.
        """
        return self.args[0]

    @property
    def sys_name(self):
        """
        The sys name that caused this error as string.
        """
        return self.args[1]

    def __str__(self):
        return 'No device {0.sys_name!r} in {0.subsystem!r}'.format(self)