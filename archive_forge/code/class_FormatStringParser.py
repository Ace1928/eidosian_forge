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
class FormatStringParser(object):
    """
    Shallow logging format string parser.

    This class enables introspection and manipulation of logging format strings
    in the three styles supported by the :mod:`logging` module starting from
    Python 3.2 (``%``, ``{`` and ``$``).
    """

    def __init__(self, style=DEFAULT_FORMAT_STYLE):
        """
        Initialize a :class:`FormatStringParser` object.

        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :raises: Refer to :func:`check_style()`.
        """
        self.style = check_style(style)
        self.capturing_pattern = FORMAT_STYLE_PATTERNS[style]
        self.raw_pattern = self.capturing_pattern.replace('(\\w+)', '\\w+')
        self.tokenize_pattern = re.compile('(%s)' % self.raw_pattern, re.VERBOSE)
        self.name_pattern = re.compile(self.capturing_pattern, re.VERBOSE)

    def contains_field(self, format_string, field_name):
        """
        Get the field names referenced by a format string.

        :param format_string: The logging format string.
        :returns: A list of strings with field names.
        """
        return field_name in self.get_field_names(format_string)

    def get_field_names(self, format_string):
        """
        Get the field names referenced by a format string.

        :param format_string: The logging format string.
        :returns: A list of strings with field names.
        """
        return self.name_pattern.findall(format_string)

    def get_grouped_pairs(self, format_string):
        """
        Group the results of :func:`get_pairs()` separated by whitespace.

        :param format_string: The logging format string.
        :returns: A list of lists of :class:`FormatStringToken` objects.
        """
        separated = []
        pattern = re.compile('(\\s+)')
        for token in self.get_pairs(format_string):
            if token.name:
                separated.append(token)
            else:
                separated.extend((FormatStringToken(name=None, text=text) for text in pattern.split(token.text) if text))
        current_group = []
        grouped_pairs = []
        for token in separated:
            if token.text.isspace():
                if current_group:
                    grouped_pairs.append(current_group)
                grouped_pairs.append([token])
                current_group = []
            else:
                current_group.append(token)
        if current_group:
            grouped_pairs.append(current_group)
        return grouped_pairs

    def get_pairs(self, format_string):
        """
        Tokenize a logging format string and extract field names from tokens.

        :param format_string: The logging format string.
        :returns: A generator of :class:`FormatStringToken` objects.
        """
        for token in self.get_tokens(format_string):
            match = self.name_pattern.search(token)
            name = match.group(1) if match else None
            yield FormatStringToken(name=name, text=token)

    def get_pattern(self, field_name):
        """
        Get a regular expression to match a formatting directive that references the given field name.

        :param field_name: The name of the field to match (a string).
        :returns: A compiled regular expression object.
        """
        return re.compile(self.raw_pattern.replace('\\w+', field_name), re.VERBOSE)

    def get_tokens(self, format_string):
        """
        Tokenize a logging format string.

        :param format_string: The logging format string.
        :returns: A list of strings with formatting directives separated from surrounding text.
        """
        return [t for t in self.tokenize_pattern.split(format_string) if t]