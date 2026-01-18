import fixtures
from debtcollector import _utils
class DisableFixture(fixtures.Fixture):
    """Fixture that disables debtcollector triggered warnings.

    This does **not** disable warnings calls emitted by other libraries.

    This can be used like::

        from debtcollector.fixtures import disable

        with disable.DisableFixture():
            <some code that calls into depreciated code>
    """

    def _setUp(self):
        self.addCleanup(setattr, _utils, '_enabled', True)
        _utils._enabled = False