import collections
import logging
import os
import re
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split
class ColoredFormatter(BasicFormatter):
    """
    Log :class:`~logging.Formatter` that uses `ANSI escape sequences`_ to create colored logs.

    :class:`ColoredFormatter` inherits from :class:`BasicFormatter` to enable
    the use of ``%f`` for millisecond formatting in date/time strings.

    .. note:: If you want to use :class:`ColoredFormatter` on Windows then you
              need to call :func:`~humanfriendly.terminal.enable_ansi_support()`.
              This is done for you when you call :func:`coloredlogs.install()`.
    """

    def __init__(self, fmt=None, datefmt=None, style=DEFAULT_FORMAT_STYLE, level_styles=None, field_styles=None):
        """
        Initialize a :class:`ColoredFormatter` object.

        :param fmt: A log format string (defaults to :data:`DEFAULT_LOG_FORMAT`).
        :param datefmt: A date/time format string (defaults to :data:`None`,
                        but see the documentation of
                        :func:`BasicFormatter.formatTime()`).
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`)
        :param level_styles: A dictionary with custom level styles
                             (defaults to :data:`DEFAULT_LEVEL_STYLES`).
        :param field_styles: A dictionary with custom field styles
                             (defaults to :data:`DEFAULT_FIELD_STYLES`).
        :raises: Refer to :func:`check_style()`.

        This initializer uses :func:`colorize_format()` to inject ANSI escape
        sequences in the log format string before it is passed to the
        initializer of the base class.
        """
        self.nn = NameNormalizer()
        fmt = fmt or DEFAULT_LOG_FORMAT
        self.level_styles = self.nn.normalize_keys(DEFAULT_LEVEL_STYLES if level_styles is None else level_styles)
        self.field_styles = self.nn.normalize_keys(DEFAULT_FIELD_STYLES if field_styles is None else field_styles)
        kw = dict(fmt=self.colorize_format(fmt, style), datefmt=datefmt)
        if style != DEFAULT_FORMAT_STYLE:
            kw['style'] = style
        logging.Formatter.__init__(self, **kw)

    def colorize_format(self, fmt, style=DEFAULT_FORMAT_STYLE):
        """
        Rewrite a logging format string to inject ANSI escape sequences.

        :param fmt: The log format string.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :returns: The logging format string with ANSI escape sequences.

        This method takes a logging format string like the ones you give to
        :class:`logging.Formatter` and processes it as follows:

        1. First the logging format string is separated into formatting
           directives versus surrounding text (according to the given `style`).

        2. Then formatting directives and surrounding text are grouped
           based on whitespace delimiters (in the surrounding text).

        3. For each group styling is selected as follows:

           1. If the group contains a single formatting directive that has
              a style defined then the whole group is styled accordingly.

           2. If the group contains multiple formatting directives that
              have styles defined then each formatting directive is styled
              individually and surrounding text isn't styled.

        As an example consider the default log format (:data:`DEFAULT_LOG_FORMAT`)::

         %(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s

        The default field styles (:data:`DEFAULT_FIELD_STYLES`) define a style for the
        `name` field but not for the `process` field, however because both fields
        are part of the same whitespace delimited token they'll be highlighted
        together in the style defined for the `name` field.
        """
        result = []
        parser = FormatStringParser(style=style)
        for group in parser.get_grouped_pairs(fmt):
            applicable_styles = [self.nn.get(self.field_styles, token.name) for token in group if token.name]
            if sum(map(bool, applicable_styles)) == 1:
                result.append(ansi_wrap(''.join((token.text for token in group)), **next((s for s in applicable_styles if s))))
            else:
                for token in group:
                    text = token.text
                    if token.name:
                        field_styles = self.nn.get(self.field_styles, token.name)
                        if field_styles:
                            text = ansi_wrap(text, **field_styles)
                    result.append(text)
        return ''.join(result)

    def format(self, record):
        """
        Apply level-specific styling to log records.

        :param record: A :class:`~logging.LogRecord` object.
        :returns: The result of :func:`logging.Formatter.format()`.

        This method injects ANSI escape sequences that are specific to the
        level of each log record (because such logic cannot be expressed in the
        syntax of a log format string). It works by making a copy of the log
        record, changing the `msg` field inside the copy and passing the copy
        into the :func:`~logging.Formatter.format()` method of the base
        class.
        """
        style = self.nn.get(self.level_styles, record.levelname)
        if style and Empty is not None:
            copy = Empty()
            copy.__class__ = record.__class__
            copy.__dict__.update(record.__dict__)
            copy.msg = ansi_wrap(coerce_string(record.msg), **style)
            record = copy
        return logging.Formatter.format(self, record)