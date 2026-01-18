import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _str_see_also(self, func_role):
    if not self['See Also']:
        return []
    out = []
    out += self._str_header('See Also')
    out += ['']
    last_had_desc = True
    for funcs, desc in self['See Also']:
        assert isinstance(funcs, list)
        links = []
        for func, role in funcs:
            if role:
                link = f':{role}:`{func}`'
            elif func_role:
                link = f':{func_role}:`{func}`'
            else:
                link = '`%s`_' % func
            links.append(link)
        link = ', '.join(links)
        out += [link]
        if desc:
            out += self._str_indent([' '.join(desc)])
            last_had_desc = True
        else:
            last_had_desc = False
            out += self._str_indent([self.empty_description])
    if last_had_desc:
        out += ['']
    out += ['']
    return out