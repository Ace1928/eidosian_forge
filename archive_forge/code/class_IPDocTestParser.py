import doctest
import logging
import re
from testpath import modified_env
class IPDocTestParser(doctest.DocTestParser):
    """
    A class used to parse strings containing doctest examples.

    Note: This is a version modified to properly recognize IPython input and
    convert any IPython examples into valid Python ones.
    """
    _PS1_PY = '>>>'
    _PS2_PY = '\\.\\.\\.'
    _PS1_IP = 'In\\ \\[\\d+\\]:'
    _PS2_IP = '\\ \\ \\ \\.\\.\\.+:'
    _RE_TPL = '\n        # Source consists of a PS1 line followed by zero or more PS2 lines.\n        (?P<source>\n            (?:^(?P<indent> [ ]*) (?P<ps1> %s) .*)    # PS1 line\n            (?:\\n           [ ]*  (?P<ps2> %s) .*)*)  # PS2 lines\n        \\n? # a newline\n        # Want consists of any non-blank lines that do not start with PS1.\n        (?P<want> (?:(?![ ]*$)    # Not a blank line\n                     (?![ ]*%s)   # Not a line starting with PS1\n                     (?![ ]*%s)   # Not a line starting with PS2\n                     .*$\\n?       # But any other line\n                  )*)\n                  '
    _EXAMPLE_RE_PY = re.compile(_RE_TPL % (_PS1_PY, _PS2_PY, _PS1_PY, _PS2_PY), re.MULTILINE | re.VERBOSE)
    _EXAMPLE_RE_IP = re.compile(_RE_TPL % (_PS1_IP, _PS2_IP, _PS1_IP, _PS2_IP), re.MULTILINE | re.VERBOSE)
    _RANDOM_TEST = re.compile('#\\s*all-random\\s+')

    def ip2py(self, source):
        """Convert input IPython source into valid Python."""
        block = _ip.input_transformer_manager.transform_cell(source)
        if len(block.splitlines()) == 1:
            return _ip.prefilter(block)
        else:
            return block

    def parse(self, string, name='<string>'):
        """
        Divide the given string into examples and intervening text,
        and return them as a list of alternating Examples and strings.
        Line numbers for the Examples are 0-based.  The optional
        argument `name` is a name identifying this string, and is only
        used for error messages.
        """
        string = string.expandtabs()
        min_indent = self._min_indent(string)
        if min_indent > 0:
            string = '\n'.join([l[min_indent:] for l in string.split('\n')])
        output = []
        charno, lineno = (0, 0)
        if self._RANDOM_TEST.search(string):
            random_marker = '\n# random'
        else:
            random_marker = ''
        ip2py = False
        terms = list(self._EXAMPLE_RE_PY.finditer(string))
        if terms:
            Example = doctest.Example
        else:
            terms = list(self._EXAMPLE_RE_IP.finditer(string))
            Example = IPExample
            ip2py = True
        for m in terms:
            output.append(string[charno:m.start()])
            lineno += string.count('\n', charno, m.start())
            source, options, want, exc_msg = self._parse_example(m, name, lineno, ip2py)
            want += random_marker
            if not self._IS_BLANK_OR_COMMENT(source):
                output.append(Example(source, want, exc_msg, lineno=lineno, indent=min_indent + len(m.group('indent')), options=options))
            lineno += string.count('\n', m.start(), m.end())
            charno = m.end()
        output.append(string[charno:])
        return output

    def _parse_example(self, m, name, lineno, ip2py=False):
        """
        Given a regular expression match from `_EXAMPLE_RE` (`m`),
        return a pair `(source, want)`, where `source` is the matched
        example's source code (with prompts and indentation stripped);
        and `want` is the example's expected output (with indentation
        stripped).

        `name` is the string's name, and `lineno` is the line number
        where the example starts; both are used for error messages.

        Optional:
        `ip2py`: if true, filter the input via IPython to convert the syntax
        into valid python.
        """
        indent = len(m.group('indent'))
        source_lines = m.group('source').split('\n')
        ps1 = m.group('ps1')
        ps2 = m.group('ps2')
        ps1_len = len(ps1)
        self._check_prompt_blank(source_lines, indent, name, lineno, ps1_len)
        if ps2:
            self._check_prefix(source_lines[1:], ' ' * indent + ps2, name, lineno)
        source = '\n'.join([sl[indent + ps1_len + 1:] for sl in source_lines])
        if ip2py:
            source = self.ip2py(source)
        want = m.group('want')
        want_lines = want.split('\n')
        if len(want_lines) > 1 and re.match(' *$', want_lines[-1]):
            del want_lines[-1]
        self._check_prefix(want_lines, ' ' * indent, name, lineno + len(source_lines))
        want_lines[0] = re.sub('Out\\[\\d+\\]: \\s*?\\n?', '', want_lines[0])
        want = '\n'.join([wl[indent:] for wl in want_lines])
        m = self._EXCEPTION_RE.match(want)
        if m:
            exc_msg = m.group('msg')
        else:
            exc_msg = None
        options = self._find_options(source, name, lineno)
        return (source, options, want, exc_msg)

    def _check_prompt_blank(self, lines, indent, name, lineno, ps1_len):
        """
        Given the lines of a source string (including prompts and
        leading indentation), check to make sure that every prompt is
        followed by a space character.  If any line is not followed by
        a space character, then raise ValueError.

        Note: IPython-modified version which takes the input prompt length as a
        parameter, so that prompts of variable length can be dealt with.
        """
        space_idx = indent + ps1_len
        min_len = space_idx + 1
        for i, line in enumerate(lines):
            if len(line) >= min_len and line[space_idx] != ' ':
                raise ValueError('line %r of the docstring for %s lacks blank after %s: %r' % (lineno + i + 1, name, line[indent:space_idx], line))