import re
from mako import exceptions
class PythonPrinter:

    def __init__(self, stream):
        self.indent = 0
        self.indent_detail = []
        self.indentstring = '    '
        self.stream = stream
        self.lineno = 1
        self.line_buffer = []
        self.in_indent_lines = False
        self._reset_multi_line_flags()
        self.source_map = {}
        self._re_space_comment = re.compile('^\\s*#')
        self._re_space = re.compile('^\\s*$')
        self._re_indent = re.compile(':[ \\t]*(?:#.*)?$')
        self._re_compound = re.compile('^\\s*(if|try|elif|while|for|with)')
        self._re_indent_keyword = re.compile('^\\s*(def|class|else|elif|except|finally)')
        self._re_unindentor = re.compile('^\\s*(else|elif|except|finally).*\\:')

    def _update_lineno(self, num):
        self.lineno += num

    def start_source(self, lineno):
        if self.lineno not in self.source_map:
            self.source_map[self.lineno] = lineno

    def write_blanks(self, num):
        self.stream.write('\n' * num)
        self._update_lineno(num)

    def write_indented_block(self, block, starting_lineno=None):
        """print a line or lines of python which already contain indentation.

        The indentation of the total block of lines will be adjusted to that of
        the current indent level."""
        self.in_indent_lines = False
        for i, l in enumerate(re.split('\\r?\\n', block)):
            self.line_buffer.append(l)
            if starting_lineno is not None:
                self.start_source(starting_lineno + i)
            self._update_lineno(1)

    def writelines(self, *lines):
        """print a series of lines of python."""
        for line in lines:
            self.writeline(line)

    def writeline(self, line):
        """print a line of python, indenting it according to the current
        indent level.

        this also adjusts the indentation counter according to the
        content of the line.

        """
        if not self.in_indent_lines:
            self._flush_adjusted_lines()
            self.in_indent_lines = True
        if line is None or self._re_space_comment.match(line) or self._re_space.match(line):
            hastext = False
        else:
            hastext = True
        is_comment = line and len(line) and (line[0] == '#')
        if not is_comment and (not hastext or self._is_unindentor(line)) and (self.indent > 0):
            self.indent -= 1
            if len(self.indent_detail) == 0:
                raise exceptions.MakoException('Too many whitespace closures')
            self.indent_detail.pop()
        if line is None:
            return
        self.stream.write(self._indent_line(line) + '\n')
        self._update_lineno(len(line.split('\n')))
        if self._re_indent.search(line):
            match = self._re_compound.match(line)
            if match:
                indentor = match.group(1)
                self.indent += 1
                self.indent_detail.append(indentor)
            else:
                indentor = None
                m2 = self._re_indent_keyword.match(line)
                if m2:
                    self.indent += 1
                    self.indent_detail.append(indentor)

    def close(self):
        """close this printer, flushing any remaining lines."""
        self._flush_adjusted_lines()

    def _is_unindentor(self, line):
        """return true if the given line is an 'unindentor',
        relative to the last 'indent' event received.

        """
        if len(self.indent_detail) == 0:
            return False
        indentor = self.indent_detail[-1]
        if indentor is None:
            return False
        match = self._re_unindentor.match(line)
        return bool(match)

    def _indent_line(self, line, stripspace=''):
        """indent the given line according to the current indent level.

        stripspace is a string of space that will be truncated from the
        start of the line before indenting."""
        if stripspace == '':
            return self.indentstring * self.indent + line
        return re.sub('^%s' % stripspace, self.indentstring * self.indent, line)

    def _reset_multi_line_flags(self):
        """reset the flags which would indicate we are in a backslashed
        or triple-quoted section."""
        self.backslashed, self.triplequoted = (False, False)

    def _in_multi_line(self, line):
        """return true if the given line is part of a multi-line block,
        via backslash or triple-quote."""
        current_state = self.backslashed or self.triplequoted
        self.backslashed = bool(re.search('\\\\$', line))
        triples = len(re.findall('\\"\\"\\"|\\\'\\\'\\\'', line))
        if triples == 1 or triples % 2 != 0:
            self.triplequoted = not self.triplequoted
        return current_state

    def _flush_adjusted_lines(self):
        stripspace = None
        self._reset_multi_line_flags()
        for entry in self.line_buffer:
            if self._in_multi_line(entry):
                self.stream.write(entry + '\n')
            else:
                entry = entry.expandtabs()
                if stripspace is None and re.search('^[ \\t]*[^# \\t]', entry):
                    stripspace = re.match('^([ \\t]*)', entry).group(1)
                self.stream.write(self._indent_line(entry, stripspace) + '\n')
        self.line_buffer = []
        self._reset_multi_line_flags()