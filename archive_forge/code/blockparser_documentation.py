from __future__ import annotations
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util
 Process blocks of Markdown text and attach to given `etree` node.

        Given a list of `blocks`, each `blockprocessor` is stepped through
        until there are no blocks left. While an extension could potentially
        call this method directly, it's generally expected to be used
        internally.

        This is a public method as an extension may need to add/alter
        additional `BlockProcessors` which call this method to recursively
        parse a nested block.

        Arguments:
            parent: The parent element.
            blocks: The blocks of text to parse.

        