import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class Section:
    """Numpydoc section parser.

    :param title: section title. For most sections, this is a heading like
                  "Parameters" which appears on its own line, underlined by
                  en-dashes ('-') on the following line.
    :param key: meta key string. In the parsed ``DocstringMeta`` instance this
                will be the first element of the ``args`` attribute list.
    """

    def __init__(self, title: str, key: str) -> None:
        self.title = title
        self.key = key

    @property
    def title_pattern(self) -> str:
        """Regular expression pattern matching this section's header.

        This pattern will match this instance's ``title`` attribute in
        an anonymous group.
        """
        dashes = '-' * len(self.title)
        return f'^({self.title})\\s*?\\n{dashes}\\s*$'

    def parse(self, text: str) -> T.Iterable[DocstringMeta]:
        """Parse ``DocstringMeta`` objects from the body of this section.

        :param text: section body text. Should be cleaned with
                     ``inspect.cleandoc`` before parsing.
        """
        yield DocstringMeta([self.key], description=_clean_str(text))