import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
class _DumbXMLWriter:

    def __init__(self, file, indent_level=0, indent='\t'):
        self.file = file
        self.stack = []
        self._indent_level = indent_level
        self.indent = indent

    def begin_element(self, element):
        self.stack.append(element)
        self.writeln('<%s>' % element)
        self._indent_level += 1

    def end_element(self, element):
        assert self._indent_level > 0
        assert self.stack.pop() == element
        self._indent_level -= 1
        self.writeln('</%s>' % element)

    def simple_element(self, element, value=None):
        if value is not None:
            value = _escape(value)
            self.writeln('<%s>%s</%s>' % (element, value, element))
        else:
            self.writeln('<%s/>' % element)

    def writeln(self, line):
        if line:
            if isinstance(line, str):
                line = line.encode('utf-8')
            self.file.write(self._indent_level * self.indent)
            self.file.write(line)
        self.file.write(b'\n')