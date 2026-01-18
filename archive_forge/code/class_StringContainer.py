import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class StringContainer(Container):
    """A container for a single string"""
    parsed = None

    def __init__(self):
        self.parser = StringParser()
        self.output = StringOutput()
        self.string = ''

    def process(self):
        """Replace special chars from elyxer.the contents."""
        if self.parsed:
            self.string = self.replacespecial(self.parsed)
            self.parsed = None

    def replacespecial(self, line):
        """Replace all special chars from elyxer.a line"""
        replaced = self.escape(line, EscapeConfig.entities)
        replaced = self.changeline(replaced)
        if ContainerConfig.string['startcommand'] in replaced and len(replaced) > 1:
            if self.begin:
                message = 'Unknown command at ' + str(self.begin) + ': '
            else:
                message = 'Unknown command: '
            Trace.error(message + replaced.strip())
        return replaced

    def changeline(self, line):
        line = self.escape(line, EscapeConfig.chars)
        if not ContainerConfig.string['startcommand'] in line:
            return line
        line = self.escape(line, EscapeConfig.commands)
        return line

    def extracttext(self):
        """Return all text."""
        return self.string

    def __unicode__(self):
        """Return a printable representation."""
        result = 'StringContainer'
        if self.begin:
            result += '@' + str(self.begin)
        ellipsis = '...'
        if len(self.string.strip()) <= 15:
            ellipsis = ''
        return result + ' (' + self.string.strip()[:15] + ellipsis + ')'