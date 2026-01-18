import io
from .. import ui
from ..ui import text as ui_text
class TestUIFactory(TextUIFactory):
    """A UI Factory for testing.

    Hide the progress bar but emit note()s.
    Redirect stdin.
    Allows get_password to be tested without real tty attached.

    See also CannedInputUIFactory which lets you provide programmatic input in
    a structured way.
    """

    def get_non_echoed_password(self):
        """Get password from stdin without trying to handle the echo mode"""
        password = self.stdin.readline()
        if not password:
            raise EOFError
        if password[-1] == '\n':
            password = password[:-1]
        return password

    def make_progress_view(self):
        return ui.NullProgressView()