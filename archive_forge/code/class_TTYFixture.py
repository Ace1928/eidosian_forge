import io
import os
import pty
import tempfile
import fixtures  # type: ignore
import typing
class TTYFixture(fixtures.Fixture):

    def _setUp(self) -> None:
        self._k, term = pty.openpty()
        self.stream = os.fdopen(term, 'w')
        self.addCleanup(self.stream.close)
        self.addCleanup(os.close, self._k)