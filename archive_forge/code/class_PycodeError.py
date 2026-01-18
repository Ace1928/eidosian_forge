from typing import Any, Optional
class PycodeError(Exception):
    """Pycode Python source code analyser error."""

    def __str__(self) -> str:
        res = self.args[0]
        if len(self.args) > 1:
            res += ' (exception was: %r)' % self.args[1]
        return res