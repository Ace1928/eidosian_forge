import io
import os
import pty
import tempfile
import fixtures  # type: ignore
import typing
class BufferFixture(fixtures.Fixture):

    def _setUp(self) -> None:
        self.stream: typing.TextIO = io.StringIO()
        self.addCleanup(self.stream.close)