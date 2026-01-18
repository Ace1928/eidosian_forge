import gettext
import fixtures
from oslo_i18n import _lazy
from oslo_i18n import _message
class ToggleLazy(fixtures.Fixture):
    """Fixture to toggle lazy translation on or off for a test."""

    def __init__(self, enabled):
        """Force lazy translation on or off.

        :param enabled: Flag controlling whether to enable or disable
            lazy translation, passed to :func:`~oslo_i18n.enable_lazy`.
        :type enabled: bool
        """
        super(ToggleLazy, self).__init__()
        self._enabled = enabled
        self._original_value = _lazy.USE_LAZY

    def setUp(self):
        super(ToggleLazy, self).setUp()
        self.addCleanup(self._restore_original)
        _lazy.enable_lazy(self._enabled)

    def _restore_original(self):
        _lazy.enable_lazy(self._original_value)