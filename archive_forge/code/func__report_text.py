from __future__ import annotations
import sys
from typing import Any, IO, Iterable, TYPE_CHECKING
from coverage.exceptions import ConfigError, NoDataError
from coverage.misc import human_sorted_items
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf
def _report_text(self, header: list[str], lines_values: list[list[Any]], total_line: list[Any], end_lines: list[str]) -> None:
    """Internal method that prints report data in text format.

        `header` is a list with captions.
        `lines_values` is list of lists of sortable values.
        `total_line` is a list with values of the total line.
        `end_lines` is a list of ending lines with information about skipped files.

        """
    max_name = max([len(line[0]) for line in lines_values] + [5]) + 1
    max_n = max(len(total_line[header.index('Cover')]) + 2, len(' Cover')) + 1
    max_n = max([max_n] + [len(line[header.index('Cover')]) + 2 for line in lines_values])
    formats = dict(Name='{:{name_len}}', Stmts='{:>7}', Miss='{:>7}', Branch='{:>7}', BrPart='{:>7}', Cover='{:>{n}}', Missing='{:>10}')
    header_items = [formats[item].format(item, name_len=max_name, n=max_n) for item in header]
    header_str = ''.join(header_items)
    rule = '-' * len(header_str)
    self.write(header_str)
    self.write(rule)
    formats.update(dict(Cover='{:>{n}}%'), Missing='   {:9}')
    for values in lines_values:
        line_items = [formats[item].format(str(value), name_len=max_name, n=max_n - 1) for item, value in zip(header, values)]
        self.write_items(line_items)
    if lines_values:
        self.write(rule)
    line_items = [formats[item].format(str(value), name_len=max_name, n=max_n - 1) for item, value in zip(header, total_line)]
    self.write_items(line_items)
    for end_line in end_lines:
        self.write(end_line)