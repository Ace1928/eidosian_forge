from __future__ import annotations
def _add_note(self, note: str) -> None:
    message, *rest = self.args
    self.args = (message + '\n' + note, *rest)