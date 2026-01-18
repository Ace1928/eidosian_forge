import collections
import inspect
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints
def _parse_numpydoc_see_also_section(self, content: List[str]) -> List[str]:
    """
        Derived from the NumpyDoc implementation of _parse_see_also.

        See Also
        --------
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """
    items = []

    def parse_item_name(text: str) -> Tuple[str, str]:
        """Match ':role:`name`' or 'name'"""
        m = self._name_rgx.match(text)
        if m:
            g = m.groups()
            if g[1] is None:
                return (g[3], None)
            else:
                return (g[2], g[1])
        raise ValueError('%s is not a item name' % text)

    def push_item(name: str, rest: List[str]) -> None:
        if not name:
            return
        name, role = parse_item_name(name)
        items.append((name, list(rest), role))
        del rest[:]

    def translate(func, description, role):
        translations = self._config.napoleon_type_aliases
        if role is not None or not translations:
            return (func, description, role)
        translated = translations.get(func, func)
        match = self._name_rgx.match(translated)
        if not match:
            return (translated, description, role)
        groups = match.groupdict()
        role = groups['role']
        new_func = groups['name'] or groups['name2']
        return (new_func, description, role)
    current_func = None
    rest: List[str] = []
    for line in content:
        if not line.strip():
            continue
        m = self._name_rgx.match(line)
        if m and line[m.end():].strip().startswith(':'):
            push_item(current_func, rest)
            current_func, line = (line[:m.end()], line[m.end():])
            rest = [line.split(':', 1)[1].strip()]
            if not rest[0]:
                rest = []
        elif not line.startswith(' '):
            push_item(current_func, rest)
            current_func = None
            if ',' in line:
                for func in line.split(','):
                    if func.strip():
                        push_item(func, [])
            elif line.strip():
                current_func = line
        elif current_func is not None:
            rest.append(line.strip())
    push_item(current_func, rest)
    if not items:
        return []
    items = [translate(func, description, role) for func, description, role in items]
    lines: List[str] = []
    last_had_desc = True
    for name, desc, role in items:
        if role:
            link = ':%s:`%s`' % (role, name)
        else:
            link = ':obj:`%s`' % name
        if desc or last_had_desc:
            lines += ['']
            lines += [link]
        else:
            lines[-1] += ', %s' % link
        if desc:
            lines += self._indent([' '.join(desc)])
            last_had_desc = True
        else:
            last_had_desc = False
    lines += ['']
    return self._format_admonition('seealso', lines)