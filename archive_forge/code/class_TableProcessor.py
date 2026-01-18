from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING, Any, Sequence
class TableProcessor(BlockProcessor):
    """ Process Tables. """
    RE_CODE_PIPES = re.compile('(?:(\\\\\\\\)|(\\\\`+)|(`+)|(\\\\\\|)|(\\|))')
    RE_END_BORDER = re.compile('(?<!\\\\)(?:\\\\\\\\)*\\|$')

    def __init__(self, parser: blockparser.BlockParser, config: dict[str, Any]):
        self.border: bool | int = False
        self.separator: Sequence[str] = ''
        self.config = config
        super().__init__(parser)

    def test(self, parent: etree.Element, block: str) -> bool:
        """
        Ensure first two rows (column header and separator row) are valid table rows.

        Keep border check and separator row do avoid repeating the work.
        """
        is_table = False
        rows = [row.strip(' ') for row in block.split('\n')]
        if len(rows) > 1:
            header0 = rows[0]
            self.border = PIPE_NONE
            if header0.startswith('|'):
                self.border |= PIPE_LEFT
            if self.RE_END_BORDER.search(header0) is not None:
                self.border |= PIPE_RIGHT
            row = self._split_row(header0)
            row0_len = len(row)
            is_table = row0_len > 1
            if not is_table and row0_len == 1 and self.border:
                for index in range(1, len(rows)):
                    is_table = rows[index].startswith('|')
                    if not is_table:
                        is_table = self.RE_END_BORDER.search(rows[index]) is not None
                    if not is_table:
                        break
            if is_table:
                row = self._split_row(rows[1])
                is_table = len(row) == row0_len and set(''.join(row)) <= set('|:- ')
                if is_table:
                    self.separator = row
        return is_table

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        """ Parse a table block and build table. """
        block = blocks.pop(0).split('\n')
        header = block[0].strip(' ')
        rows = [] if len(block) < 3 else block[2:]
        align: list[str | None] = []
        for c in self.separator:
            c = c.strip(' ')
            if c.startswith(':') and c.endswith(':'):
                align.append('center')
            elif c.startswith(':'):
                align.append('left')
            elif c.endswith(':'):
                align.append('right')
            else:
                align.append(None)
        table = etree.SubElement(parent, 'table')
        thead = etree.SubElement(table, 'thead')
        self._build_row(header, thead, align)
        tbody = etree.SubElement(table, 'tbody')
        if len(rows) == 0:
            self._build_empty_row(tbody, align)
        else:
            for row in rows:
                self._build_row(row.strip(' '), tbody, align)

    def _build_empty_row(self, parent: etree.Element, align: Sequence[str | None]) -> None:
        """Build an empty row."""
        tr = etree.SubElement(parent, 'tr')
        count = len(align)
        while count:
            etree.SubElement(tr, 'td')
            count -= 1

    def _build_row(self, row: str, parent: etree.Element, align: Sequence[str | None]) -> None:
        """ Given a row of text, build table cells. """
        tr = etree.SubElement(parent, 'tr')
        tag = 'td'
        if parent.tag == 'thead':
            tag = 'th'
        cells = self._split_row(row)
        for i, a in enumerate(align):
            c = etree.SubElement(tr, tag)
            try:
                c.text = cells[i].strip(' ')
            except IndexError:
                c.text = ''
            if a:
                if self.config['use_align_attribute']:
                    c.set('align', a)
                else:
                    c.set('style', f'text-align: {a};')

    def _split_row(self, row: str) -> list[str]:
        """ split a row of text into list of cells. """
        if self.border:
            if row.startswith('|'):
                row = row[1:]
            row = self.RE_END_BORDER.sub('', row)
        return self._split(row)

    def _split(self, row: str) -> list[str]:
        """ split a row of text with some code into a list of cells. """
        elements = []
        pipes = []
        tics = []
        tic_points = []
        tic_region = []
        good_pipes = []
        for m in self.RE_CODE_PIPES.finditer(row):
            if m.group(2):
                tics.append(len(m.group(2)) - 1)
                tic_points.append((m.start(2), m.end(2) - 1, 1))
            elif m.group(3):
                tics.append(len(m.group(3)))
                tic_points.append((m.start(3), m.end(3) - 1, 0))
            elif m.group(5):
                pipes.append(m.start(5))
        pos = 0
        tic_len = len(tics)
        while pos < tic_len:
            try:
                tic_size = tics[pos] - tic_points[pos][2]
                if tic_size == 0:
                    raise ValueError
                index = tics[pos + 1:].index(tic_size) + 1
                tic_region.append((tic_points[pos][0], tic_points[pos + index][1]))
                pos += index + 1
            except ValueError:
                pos += 1
        for pipe in pipes:
            throw_out = False
            for region in tic_region:
                if pipe < region[0]:
                    break
                elif region[0] <= pipe <= region[1]:
                    throw_out = True
                    break
            if not throw_out:
                good_pipes.append(pipe)
        pos = 0
        for pipe in good_pipes:
            elements.append(row[pos:pipe])
            pos = pipe + 1
        elements.append(row[pos:])
        return elements