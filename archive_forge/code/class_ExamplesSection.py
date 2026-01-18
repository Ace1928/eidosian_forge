import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class ExamplesSection(Section):
    """Parser for numpydoc examples sections.

    E.g. any section that looks like this:
        >>> import numpy.matlib
        >>> np.matlib.empty((2, 2))    # filled with random data
        matrix([[  6.76425276e-320,   9.79033856e-307], # random
                [  7.39337286e-309,   3.22135945e-309]])
        >>> np.matlib.empty((2, 2), dtype=int)
        matrix([[ 6600475,        0], # random
                [ 6586976, 22740995]])
    """

    def parse(self, text: str) -> T.Iterable[DocstringMeta]:
        """Parse ``DocstringExample`` objects from the body of this section.

        :param text: section body text. Should be cleaned with
                     ``inspect.cleandoc`` before parsing.
        """
        lines = dedent(text).strip().splitlines()
        while lines:
            snippet_lines = []
            description_lines = []
            while lines:
                if not lines[0].startswith('>>>'):
                    break
                snippet_lines.append(lines.pop(0))
            while lines:
                if lines[0].startswith('>>>'):
                    break
                description_lines.append(lines.pop(0))
            yield DocstringExample([self.key], snippet='\n'.join(snippet_lines) if snippet_lines else None, description='\n'.join(description_lines))