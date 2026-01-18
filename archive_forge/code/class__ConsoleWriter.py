from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
class _ConsoleWriter(object):
    """A class that wraps stdout or stderr so we can control how it gets logged.

  This class is a stripped down file-like object that provides the basic
  writing methods.  When you write to this stream, if it is enabled, it will be
  written to stdout.  All strings will also be logged at DEBUG level so they
  can be captured by the log file.
  """

    def __init__(self, logger, output_filter, stream_wrapper, always_flush=False):
        """Creates a new _ConsoleWriter wrapper.

    Args:
      logger: logging.Logger, The logger to log to.
      output_filter: _UserOutputFilter, Used to determine whether to write
        output or not.
      stream_wrapper: _StreamWrapper, The wrapper for the output stream,
        stdout or stderr.
      always_flush: bool, always flush stream_wrapper, default to False.
    """
        self.__logger = logger
        self.__filter = output_filter
        self.__stream_wrapper = stream_wrapper
        self.__always_flush = always_flush

    def ParseMsg(self, msg):
        """Converts msg to a console safe pair of plain and ANSI-annotated strings.

    Args:
      msg: str or text.TypedText, the message to parse into plain and
        ANSI-annotated strings.
    Returns:
      str, str: A plain text string and a string that may also contain ANSI
        constrol sequences. If ANSI is not supported or color is disabled,
        then the second string will be identical to the first.
    """
        plain_text, styled_text = (msg, msg)
        if isinstance(msg, text.TypedText):
            typed_text_parser = style_parser.GetTypedTextParser()
            plain_text = typed_text_parser.ParseTypedTextToString(msg, stylize=False)
            styled_text = typed_text_parser.ParseTypedTextToString(msg, stylize=self.isatty())
        plain_text = console_attr.SafeText(plain_text, encoding=LOG_FILE_ENCODING, escape=False)
        styled_text = console_attr.SafeText(styled_text, encoding=LOG_FILE_ENCODING, escape=False)
        return (plain_text, styled_text)

    def Print(self, *tokens):
        """Writes the given tokens to the output stream, and adds a newline.

    This method has the same output behavior as the builtin print method but
    respects the configured verbosity.

    Args:
      *tokens: str or text.TypedTextor any object with a str() or unicode()
        method, The messages to print, which are joined with ' '.
    """
        plain_tokens, styled_tokens = ([], [])
        for token in tokens:
            plain_text, styled_text = self.ParseMsg(token)
            plain_tokens.append(plain_text)
            styled_tokens.append(styled_text)
        plain_text = ' '.join(plain_tokens) + '\n'
        styled_text = ' '.join(styled_tokens) + '\n'
        self._Write(plain_text, styled_text)

    def GetConsoleWriterStream(self):
        """Returns the console writer output stream."""
        return self.__stream_wrapper.stream

    def _Write(self, msg, styled_msg):
        """Just a helper so we don't have to double encode from Print and write.

    Args:
      msg: A text string that only has characters that are safe to encode with
        utf-8.
      styled_msg: A text string with the same properties as msg but also
        contains ANSI control sequences.
    """
        self.__logger.info(msg)
        if self.__filter.enabled:
            stream_encoding = console_attr.GetConsoleAttr().GetEncoding()
            stream_msg = console_attr.SafeText(styled_msg, encoding=stream_encoding, escape=False)
            if six.PY2:
                stream_msg = styled_msg.encode(stream_encoding or 'utf-8', 'replace')
            self.__stream_wrapper.stream.write(stream_msg)
            if self.__always_flush:
                self.flush()

    def write(self, msg):
        plain_text, styled_text = self.ParseMsg(msg)
        self._Write(plain_text, styled_text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        if self.__filter.enabled:
            self.__stream_wrapper.stream.flush()

    def isatty(self):
        isatty = getattr(self.__stream_wrapper.stream, 'isatty', None)
        return isatty() if isatty else False