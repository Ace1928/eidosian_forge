from __future__ import annotations
from . import Extension
from ..preprocessors import Preprocessor
import re
import logging
from typing import Any
class MetaPreprocessor(Preprocessor):
    """ Get Meta-Data. """

    def run(self, lines: list[str]) -> list[str]:
        """ Parse Meta-Data and store in Markdown.Meta. """
        meta: dict[str, Any] = {}
        key = None
        if lines and BEGIN_RE.match(lines[0]):
            lines.pop(0)
        while lines:
            line = lines.pop(0)
            m1 = META_RE.match(line)
            if line.strip() == '' or END_RE.match(line):
                break
            if m1:
                key = m1.group('key').lower().strip()
                value = m1.group('value').strip()
                try:
                    meta[key].append(value)
                except KeyError:
                    meta[key] = [value]
            else:
                m2 = META_MORE_RE.match(line)
                if m2 and key:
                    meta[key].append(m2.group('value').strip())
                else:
                    lines.insert(0, line)
                    break
        self.md.Meta = meta
        return lines