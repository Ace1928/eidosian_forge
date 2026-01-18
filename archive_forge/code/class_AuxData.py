from __future__ import unicode_literals, with_statement
import re
import pybtex.io
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
from pybtex import py3compat
class AuxData(object):
    command_re = re.compile('\\\\(citation|bibdata|bibstyle|@input){(.*)}')
    context = None
    style = None
    data = None
    citations = None

    def __init__(self, encoding):
        self.encoding = encoding
        self.citations = []
        self._canonical_keys = {}

    def handle_citation(self, keys):
        for key in keys.split(','):
            key_lower = key.lower()
            if key_lower in self._canonical_keys:
                existing_key = self._canonical_keys[key_lower]
                if key != existing_key:
                    msg = 'case mismatch error between cite keys {0} and {1}'
                    report_error(AuxDataError(msg.format(key, existing_key), self.context))
            self.citations.append(key)
            self._canonical_keys[key_lower] = key

    def handle_bibstyle(self, style):
        if self.style is not None:
            report_error(AuxDataError('illegal, another \\bibstyle command', self.context))
        else:
            self.style = style

    def handle_bibdata(self, bibdata):
        if self.data is not None:
            report_error(AuxDataError('illegal, another \\bibdata command', self.context))
        else:
            self.data = bibdata.split(',')

    def handle_input(self, filename):
        self.parse_file(filename, toplevel=False)

    def handle_command(self, command, value):
        action = getattr(self, 'handle_%s' % command.lstrip('@'))
        action(value)

    def parse_line(self, line, lineno):
        self.context.lineno = lineno
        self.context.line = line.strip()
        match = self.command_re.match(line)
        if match:
            command, value = match.groups()
            self.handle_command(command, value)

    def parse_file(self, filename, toplevel=True):
        previous_context = self.context
        self.context = AuxDataContext(filename)
        with pybtex.io.open_unicode(filename, encoding=self.encoding) as aux_file:
            for lineno, line in enumerate(aux_file, 1):
                self.parse_line(line, lineno)
        if previous_context:
            self.context = previous_context
        else:
            self.context.line = None
            self.context.lineno = None
        if toplevel and self.data is None:
            raise AuxDataError('found no \\bibdata command', self.context)
        if toplevel and self.style is None:
            raise AuxDataError('found no \\bibstyle command', self.context)