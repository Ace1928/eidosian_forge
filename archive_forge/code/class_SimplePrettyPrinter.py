from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
class SimplePrettyPrinter(object):
    """Simple pretty-printer that supports an indent contextmanager."""

    def __init__(self, out):
        self.__out = out
        self.__indent = ''
        self.__skip = False
        self.__comment_context = False

    @property
    def indent(self):
        return self.__indent

    def CalculateWidth(self, max_width=78):
        return max_width - len(self.indent)

    @contextlib.contextmanager
    def Indent(self, indent='  '):
        previous_indent = self.__indent
        self.__indent = '%s%s' % (previous_indent, indent)
        yield
        self.__indent = previous_indent

    @contextlib.contextmanager
    def CommentContext(self):
        """Print without any argument formatting."""
        old_context = self.__comment_context
        self.__comment_context = True
        yield
        self.__comment_context = old_context

    def __call__(self, *args):
        if self.__comment_context and args[1:]:
            raise Error('Cannot do string interpolation in comment context')
        if args and args[0]:
            if not self.__comment_context:
                line = (args[0] % args[1:]).rstrip()
            else:
                line = args[0].rstrip()
            line = ReplaceHomoglyphs(line)
            try:
                print('%s%s' % (self.__indent, line), file=self.__out)
            except UnicodeEncodeError:
                line = line.encode('ascii', 'backslashreplace').decode('ascii')
                print('%s%s' % (self.__indent, line), file=self.__out)
        else:
            print('', file=self.__out)