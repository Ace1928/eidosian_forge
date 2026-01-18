import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _parse_see_also(self, content):
    """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """
    items = []

    def parse_item_name(text):
        """Match ':role:`name`' or 'name'."""
        m = self._func_rgx.match(text)
        if not m:
            raise ParseError('%s is not a item name' % text)
        role = m.group('role')
        name = m.group('name') if role else m.group('name2')
        return (name, role, m.end())
    rest = []
    for line in content:
        if not line.strip():
            continue
        line_match = self._line_rgx.match(line)
        description = None
        if line_match:
            description = line_match.group('desc')
            if line_match.group('trailing') and description:
                self._error_location('Unexpected comma or period after function list at index %d of line "%s"' % (line_match.end('trailing'), line), error=False)
        if not description and line.startswith(' '):
            rest.append(line.strip())
        elif line_match:
            funcs = []
            text = line_match.group('allfuncs')
            while True:
                if not text.strip():
                    break
                name, role, match_end = parse_item_name(text)
                funcs.append((name, role))
                text = text[match_end:].strip()
                if text and text[0] == ',':
                    text = text[1:].strip()
            rest = list(filter(None, [description]))
            items.append((funcs, rest))
        else:
            raise ParseError('%s is not a item name' % line)
    return items