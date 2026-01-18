import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
class NumpyDocString(Mapping):
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.

    """
    sections = {'Signature': '', 'Summary': [''], 'Extended Summary': [], 'Parameters': [], 'Returns': [], 'Yields': [], 'Receives': [], 'Raises': [], 'Warns': [], 'Other Parameters': [], 'Attributes': [], 'Methods': [], 'See Also': [], 'Notes': [], 'Warnings': [], 'References': '', 'Examples': '', 'index': {}}

    def __init__(self, docstring, config={}):
        orig_docstring = docstring
        docstring = textwrap.dedent(docstring).split('\n')
        self._doc = Reader(docstring)
        self._parsed_data = copy.deepcopy(self.sections)
        try:
            self._parse()
        except ParseError as e:
            e.docstring = orig_docstring
            raise

    def __getitem__(self, key):
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            self._error_location('Unknown section %s' % key, error=False)
        else:
            self._parsed_data[key] = val

    def __iter__(self):
        return iter(self._parsed_data)

    def __len__(self):
        return len(self._parsed_data)

    def _is_at_section(self):
        self._doc.seek_next_non_empty_line()
        if self._doc.eof():
            return False
        l1 = self._doc.peek().strip()
        if l1.startswith('.. index::'):
            return True
        l2 = self._doc.peek(1).strip()
        return l2.startswith('-' * len(l1)) or l2.startswith('=' * len(l1))

    def _strip(self, doc):
        i = 0
        j = 0
        for i, line in enumerate(doc):
            if line.strip():
                break
        for j, line in enumerate(doc[::-1]):
            if line.strip():
                break
        return doc[i:len(doc) - j]

    def _read_to_next_section(self):
        section = self._doc.read_to_next_empty_line()
        while not self._is_at_section() and (not self._doc.eof()):
            if not self._doc.peek(-1).strip():
                section += ['']
            section += self._doc.read_to_next_empty_line()
        return section

    def _read_sections(self):
        while not self._doc.eof():
            data = self._read_to_next_section()
            name = data[0].strip()
            if name.startswith('..'):
                yield (name, data[1:])
            elif len(data) < 2:
                yield StopIteration
            else:
                yield (name, self._strip(data[2:]))

    def _parse_param_list(self, content, single_element_is_type=False):
        r = Reader(content)
        params = []
        while not r.eof():
            header = r.read().strip()
            if ' : ' in header:
                arg_name, arg_type = header.split(' : ')[:2]
            elif single_element_is_type:
                arg_name, arg_type = ('', header)
            else:
                arg_name, arg_type = (header, '')
            desc = r.read_to_next_unindented_line()
            desc = dedent_lines(desc)
            desc = strip_blank_lines(desc)
            params.append(Parameter(arg_name, arg_type, desc))
        return params
    _role = ':(?P<role>\\w+):'
    _funcbacktick = '`(?P<name>(?:~\\w+\\.)?[a-zA-Z0-9_\\.-]+)`'
    _funcplain = '(?P<name2>[a-zA-Z0-9_\\.-]+)'
    _funcname = '(' + _role + _funcbacktick + '|' + _funcplain + ')'
    _funcnamenext = _funcname.replace('role', 'rolenext')
    _funcnamenext = _funcnamenext.replace('name', 'namenext')
    _description = '(?P<description>\\s*:(\\s+(?P<desc>\\S+.*))?)?\\s*$'
    _func_rgx = re.compile('^\\s*' + _funcname + '\\s*')
    _line_rgx = re.compile('^\\s*' + '(?P<allfuncs>' + _funcname + '(?P<morefuncs>([,]\\s+' + _funcnamenext + ')*)' + ')' + '(?P<trailing>[,\\.])?' + _description)
    empty_description = '..'

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

    def _parse_index(self, section, content):
        """
        .. index:: default
           :refguide: something, else, and more

        """

        def strip_each_in(lst):
            return [s.strip() for s in lst]
        out = {}
        section = section.split('::')
        if len(section) > 1:
            out['default'] = strip_each_in(section[1].split(','))[0]
        for line in content:
            line = line.split(':')
            if len(line) > 2:
                out[line[1]] = strip_each_in(line[2].split(','))
        return out

    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        if self._is_at_section():
            return
        while True:
            summary = self._doc.read_to_next_empty_line()
            summary_str = ' '.join([s.strip() for s in summary]).strip()
            compiled = re.compile('^([\\w., ]+=)?\\s*[\\w\\.]+\\(.*\\)$')
            if compiled.match(summary_str):
                self['Signature'] = summary_str
                if not self._is_at_section():
                    continue
            break
        if summary is not None:
            self['Summary'] = summary
        if not self._is_at_section():
            self['Extended Summary'] = self._read_to_next_section()

    def _parse(self):
        self._doc.reset()
        self._parse_summary()
        sections = list(self._read_sections())
        section_names = {section for section, content in sections}
        has_returns = 'Returns' in section_names
        has_yields = 'Yields' in section_names
        if has_returns and has_yields:
            msg = 'Docstring contains both a Returns and Yields section.'
            raise ValueError(msg)
        if not has_yields and 'Receives' in section_names:
            msg = 'Docstring contains a Receives section but not Yields.'
            raise ValueError(msg)
        for section, content in sections:
            if not section.startswith('..'):
                section = (s.capitalize() for s in section.split(' '))
                section = ' '.join(section)
                if self.get(section):
                    self._error_location('The section %s appears twice' % section)
            if section in ('Parameters', 'Other Parameters', 'Attributes', 'Methods'):
                self[section] = self._parse_param_list(content)
            elif section in ('Returns', 'Yields', 'Raises', 'Warns', 'Receives'):
                self[section] = self._parse_param_list(content, single_element_is_type=True)
            elif section.startswith('.. index::'):
                self['index'] = self._parse_index(section, content)
            elif section == 'See Also':
                self['See Also'] = self._parse_see_also(content)
            else:
                self[section] = content

    def _error_location(self, msg, error=True):
        if hasattr(self, '_obj'):
            try:
                filename = inspect.getsourcefile(self._obj)
            except TypeError:
                filename = None
            msg = msg + f' in the docstring of {self._obj} in {filename}.'
        if error:
            raise ValueError(msg)
        else:
            warn(msg, stacklevel=3)

    def _str_header(self, name, symbol='-'):
        return [name, len(name) * symbol]

    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [' ' * indent + line]
        return out

    def _str_signature(self):
        if self['Signature']:
            return [self['Signature'].replace('*', '\\*')] + ['']
        else:
            return ['']

    def _str_summary(self):
        if self['Summary']:
            return self['Summary'] + ['']
        else:
            return []

    def _str_extended_summary(self):
        if self['Extended Summary']:
            return self['Extended Summary'] + ['']
        else:
            return []

    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            for param in self[name]:
                parts = []
                if param.name:
                    parts.append(param.name)
                if param.type:
                    parts.append(param.type)
                out += [' : '.join(parts)]
                if param.desc and ''.join(param.desc).strip():
                    out += self._str_indent(param.desc)
            out += ['']
        return out

    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += ['']
        return out

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

    def _str_index(self):
        idx = self['index']
        out = []
        output_index = False
        default_index = idx.get('default', '')
        if default_index:
            output_index = True
        out += ['.. index:: %s' % default_index]
        for section, references in idx.items():
            if section == 'default':
                continue
            output_index = True
            out += ['   :{}: {}'.format(section, ', '.join(references))]
        if output_index:
            return out
        else:
            return ''

    def __str__(self, func_role=''):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in ('Parameters', 'Returns', 'Yields', 'Receives', 'Other Parameters', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        out += self._str_section('Warnings')
        out += self._str_see_also(func_role)
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        for param_list in ('Attributes', 'Methods'):
            out += self._str_param_list(param_list)
        out += self._str_index()
        return '\n'.join(out)