import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
class PackageFile:
    """A Debian package file.

    Objects of this class can be used to read Debian's Source and
    Packages files."""
    re_field = re.compile('^([A-Za-z][A-Za-z0-9-_]+):(?:\\s*(.*?))?\\s*$')
    re_continuation = re.compile('^\\s+(?:\\.|(\\S.*?)\\s*)$')

    def __init__(self, name, file_obj=None, encoding='utf-8'):
        """Creates a new package file object.

        name - the name of the file the data comes from
        file_obj - an alternate data source; the default is to open the
                  file with the indicated name.
        """
        if file_obj is None:
            file_obj = open(name, 'rb')
        self.name = name
        self.file = file_obj
        self.lineno = 0
        self.encoding = encoding

    def __iter__(self):
        line = self._aux_read_line()
        self.lineno += 1
        pkg = []
        while line:
            if line.strip(' \t') == '\n':
                if not pkg:
                    self.raise_syntax_error('expected package record')
                yield pkg
                pkg = []
                line = self._aux_read_line()
                self.lineno += 1
                continue
            match = self.re_field.match(line)
            if not match:
                self.raise_syntax_error('expected package field')
            name, contents = match.groups()
            contents = contents or ''
            while True:
                line = self._aux_read_line()
                self.lineno += 1
                match = self.re_continuation.match(line)
                if match:
                    ncontents, = match.groups()
                    if ncontents is None:
                        ncontents = ''
                    contents = '%s\n%s' % (contents, ncontents)
                else:
                    break
            pkg.append((name, contents))
        if pkg:
            yield pkg

    def _aux_read_line(self):
        line = self.file.readline()
        if isinstance(line, bytes):
            return line.decode(self.encoding)
        return line

    def raise_syntax_error(self, msg, lineno=None):
        if lineno is None:
            lineno = self.lineno
        raise ParseError(self.name, lineno, msg)
    raiseSyntaxError = function_deprecated_by(raise_syntax_error)