from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class BlockProcessor:
    """ Base class for block processors.

    Each subclass will provide the methods below to work with the source and
    tree. Each processor will need to define it's own `test` and `run`
    methods. The `test` method should return True or False, to indicate
    whether the current block should be processed by this processor. If the
    test passes, the parser will call the processors `run` method.

    Attributes:
        BlockProcessor.parser (BlockParser): The `BlockParser` instance this is attached to.
        BlockProcessor.tab_length (int): The tab length set on the `Markdown` instance.

    """

    def __init__(self, parser: BlockParser):
        self.parser = parser
        self.tab_length = parser.md.tab_length

    def lastChild(self, parent: etree.Element) -> etree.Element | None:
        """ Return the last child of an `etree` element. """
        if len(parent):
            return parent[-1]
        else:
            return None

    def detab(self, text: str, length: int | None=None) -> tuple[str, str]:
        """ Remove a tab from the front of each line of the given text. """
        if length is None:
            length = self.tab_length
        newtext = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith(' ' * length):
                newtext.append(line[length:])
            elif not line.strip():
                newtext.append('')
            else:
                break
        return ('\n'.join(newtext), '\n'.join(lines[len(newtext):]))

    def looseDetab(self, text: str, level: int=1) -> str:
        """ Remove a tab from front of lines but allowing dedented lines. """
        lines = text.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith(' ' * self.tab_length * level):
                lines[i] = lines[i][self.tab_length * level:]
        return '\n'.join(lines)

    def test(self, parent: etree.Element, block: str) -> bool:
        """ Test for block type. Must be overridden by subclasses.

        As the parser loops through processors, it will call the `test`
        method on each to determine if the given block of text is of that
        type. This method must return a boolean `True` or `False`. The
        actual method of testing is left to the needs of that particular
        block type. It could be as simple as `block.startswith(some_string)`
        or a complex regular expression. As the block type may be different
        depending on the parent of the block (i.e. inside a list), the parent
        `etree` element is also provided and may be used as part of the test.

        Keyword arguments:
            parent: An `etree` element which will be the parent of the block.
            block: A block of text from the source which has been split at blank lines.
        """
        pass

    def run(self, parent: etree.Element, blocks: list[str]) -> bool | None:
        """ Run processor. Must be overridden by subclasses.

        When the parser determines the appropriate type of a block, the parser
        will call the corresponding processor's `run` method. This method
        should parse the individual lines of the block and append them to
        the `etree`.

        Note that both the `parent` and `etree` keywords are pointers
        to instances of the objects which should be edited in place. Each
        processor must make changes to the existing objects as there is no
        mechanism to return new/different objects to replace them.

        This means that this method should be adding `SubElements` or adding text
        to the parent, and should remove (`pop`) or add (`insert`) items to
        the list of blocks.

        If `False` is returned, this will have the same effect as returning `False`
        from the `test` method.

        Keyword arguments:
            parent: An `etree` element which is the parent of the current block.
            blocks: A list of all remaining blocks of the document.
        """
        pass