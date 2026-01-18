import os
import shutil
import sys
from typing import final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TextIO
from .wcwidth import wcswidth
def _write_source(self, lines: Sequence[str], indents: Sequence[str]=()) -> None:
    """Write lines of source code possibly highlighted.

        Keeping this private for now because the API is clunky. We should discuss how
        to evolve the terminal writer so we can have more precise color support, for example
        being able to write part of a line in one color and the rest in another, and so on.
        """
    if indents and len(indents) != len(lines):
        raise ValueError(f'indents size ({len(indents)}) should have same size as lines ({len(lines)})')
    if not indents:
        indents = [''] * len(lines)
    source = '\n'.join(lines)
    new_lines = self._highlight(source).splitlines()
    for indent, new_line in zip(indents, new_lines):
        self.line(indent + new_line)