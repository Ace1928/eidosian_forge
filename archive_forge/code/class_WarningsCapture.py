import warnings
import fixtures
class WarningsCapture(fixtures.Fixture):
    """Capture warnings.

    While ``WarningsCapture`` is active, warnings will be captured by
    the fixture (so that they can be later analyzed).

    :attribute captures: A list of warning capture ``WarningMessage`` objects.
    """

    def _showwarning(self, *args, **kwargs):
        self.captures.append(warnings.WarningMessage(*args, **kwargs))

    def _setUp(self):
        patch = fixtures.MonkeyPatch('warnings.showwarning', self._showwarning)
        self.useFixture(patch)
        self.captures = []