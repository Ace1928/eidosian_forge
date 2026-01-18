from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
def __findPlaceholder(self, data: str, index: int) -> tuple[str | None, int]:
    """
        Extract id from data string, start from index.

        Arguments:
            data: String.
            index: Index, from which we start search.

        Returns:
            Placeholder id and string index, after the found placeholder.

        """
    m = self.__placeholder_re.search(data, index)
    if m:
        return (m.group(1), m.end())
    else:
        return (None, index + 1)