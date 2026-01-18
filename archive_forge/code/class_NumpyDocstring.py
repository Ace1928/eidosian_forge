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
class NumpyDocstring(GoogleDocstring):
    """Convert NumPy style docstrings to reStructuredText.

    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.
    config: :obj:`sphinx.ext.napoleon.Config` or :obj:`sphinx.config.Config`
        The configuration settings to use. If not given, defaults to the
        config object on `app`; or if `app` is not given defaults to the
        a new :class:`sphinx.ext.napoleon.Config` object.


    Other Parameters
    ----------------
    app : :class:`sphinx.application.Sphinx`, optional
        Application object representing the Sphinx process.
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : :class:`sphinx.ext.autodoc.Options`, optional
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.


    Example
    -------
    >>> from sphinx.ext.napoleon import Config
    >>> config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    >>> docstring = '''One line summary.
    ...
    ... Extended description.
    ...
    ... Parameters
    ... ----------
    ... arg1 : int
    ...     Description of `arg1`
    ... arg2 : str
    ...     Description of `arg2`
    ... Returns
    ... -------
    ... str
    ...     Description of return value.
    ... '''
    >>> print(NumpyDocstring(docstring, config))
    One line summary.
    <BLANKLINE>
    Extended description.
    <BLANKLINE>
    :param arg1: Description of `arg1`
    :type arg1: int
    :param arg2: Description of `arg2`
    :type arg2: str
    <BLANKLINE>
    :returns: Description of return value.
    :rtype: str
    <BLANKLINE>

    Methods
    -------
    __str__()
        Return the parsed docstring in reStructuredText format.

        Returns
        -------
        str
            UTF-8 encoded version of the docstring.

    __unicode__()
        Return the parsed docstring in reStructuredText format.

        Returns
        -------
        unicode
            Unicode version of the docstring.

    lines()
        Return the parsed lines of the docstring in reStructuredText format.

        Returns
        -------
        list(str)
            The lines of the docstring in a list.

    """

    def __init__(self, docstring: Union[str, List[str]], config: SphinxConfig=None, app: Sphinx=None, what: str='', name: str='', obj: Any=None, options: Any=None) -> None:
        self._directive_sections = ['.. index::']
        super().__init__(docstring, config, app, what, name, obj, options)

    def _get_location(self) -> str:
        try:
            filepath = inspect.getfile(self._obj) if self._obj is not None else None
        except TypeError:
            filepath = None
        name = self._name
        if filepath is None and name is None:
            return None
        elif filepath is None:
            filepath = ''
        return ':'.join([filepath, 'docstring of %s' % name])

    def _escape_args_and_kwargs(self, name: str) -> str:
        func = super()._escape_args_and_kwargs
        if ', ' in name:
            return ', '.join((func(param) for param in name.split(', ')))
        else:
            return func(name)

    def _consume_field(self, parse_type: bool=True, prefer_type: bool=False) -> Tuple[str, str, List[str]]:
        line = self._lines.next()
        if parse_type:
            _name, _, _type = self._partition_field_on_colon(line)
        else:
            _name, _type = (line, '')
        _name, _type = (_name.strip(), _type.strip())
        _name = self._escape_args_and_kwargs(_name)
        if parse_type and (not _type):
            _type = self._lookup_annotation(_name)
        if prefer_type and (not _type):
            _type, _name = (_name, _type)
        if self._config.napoleon_preprocess_types:
            _type = _convert_numpy_type_spec(_type, location=self._get_location(), translations=self._config.napoleon_type_aliases or {})
        indent = self._get_indent(line) + 1
        _desc = self._dedent(self._consume_indented_block(indent))
        _desc = self.__class__(_desc, self._config).lines()
        return (_name, _type, _desc)

    def _consume_returns_section(self, preprocess_types: bool=False) -> List[Tuple[str, str, List[str]]]:
        return self._consume_fields(prefer_type=True)

    def _consume_section_header(self) -> str:
        section = self._lines.next()
        if not _directive_regex.match(section):
            self._lines.next()
        return section

    def _is_section_break(self) -> bool:
        line1, line2 = (self._lines.get(0), self._lines.get(1))
        return not self._lines or self._is_section_header() or ['', ''] == [line1, line2] or (self._is_in_section and line1 and (not self._is_indented(line1, self._section_indent)))

    def _is_section_header(self) -> bool:
        section, underline = (self._lines.get(0), self._lines.get(1))
        section = section.lower()
        if section in self._sections and isinstance(underline, str):
            return bool(_numpy_section_regex.match(underline))
        elif self._directive_sections:
            if _directive_regex.match(section):
                for directive_section in self._directive_sections:
                    if section.startswith(directive_section):
                        return True
        return False

    def _parse_see_also_section(self, section: str) -> List[str]:
        lines = self._consume_to_next_section()
        try:
            return self._parse_numpydoc_see_also_section(lines)
        except ValueError:
            return self._format_admonition('seealso', lines)

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