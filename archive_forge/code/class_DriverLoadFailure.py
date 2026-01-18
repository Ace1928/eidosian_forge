from zaqarclient._i18n import _  # noqa
class DriverLoadFailure(ZaqarError):
    """Raised if a transport driver can't be loaded."""

    def __init__(self, driver, ex):
        msg = _('Failed to load transport driver "%(driver)s": %(error)s') % {'driver': driver, 'error': ex}
        super(DriverLoadFailure, self).__init__(msg)
        self.driver = driver
        self.ex = ex