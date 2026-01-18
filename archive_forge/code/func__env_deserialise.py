import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@staticmethod
def _env_deserialise(serialised):
    """ extract the environment variables and values from the text

        Format is:
            VAR_NAME="value"

        with ignorable whitespace around the construct (and separating each
        item). Quote characters within the value are backslash escaped.

        When producing the buildinfo file, dpkg only includes specifically
        allowed environment variables and thus there is no defined quoting
        rules for the variable names.

        The format is described by deb-buildinfo(5) and implemented in
        dpkg source scripts/dpkg-genbuildinfo.pl:cleansed_environment(),
        while the environment variables that are included in the output are
        listed in dpkg source scripts/Dpkg/Build/Info.pm
        """
    state = BuildInfo._EnvParserState.IGNORE_WHITESPACE
    name = ''
    value = None
    for ch in serialised:
        if state == BuildInfo._EnvParserState.IGNORE_WHITESPACE:
            if not ch.isspace():
                state = BuildInfo._EnvParserState.VAR_NAME
                name = ch
            continue
        if state == BuildInfo._EnvParserState.VAR_NAME:
            if ch != '=':
                name += ch
            else:
                state = BuildInfo._EnvParserState.START_VALUE_QUOTE
                value = ''
            continue
        if state == BuildInfo._EnvParserState.START_VALUE_QUOTE:
            if ch == '"':
                state = BuildInfo._EnvParserState.VALUE
            else:
                raise ValueError('Improper quoting in Environment: begin quote not found')
            continue
        if state == BuildInfo._EnvParserState.VALUE:
            if ch == '\\':
                state = BuildInfo._EnvParserState.VALUE_BACKSLASH_ESCAPE
            elif ch == '"':
                if name == '':
                    raise ValueError('Improper formatting in Environment: variable name not found')
                if value is None:
                    raise ValueError('Improper formatting in Environment: variable value not found')
                yield (name, value)
                state = BuildInfo._EnvParserState.IGNORE_WHITESPACE
                name = ''
                value = None
            else:
                assert value is not None
                value += ch
            continue
        if state == BuildInfo._EnvParserState.VALUE_BACKSLASH_ESCAPE:
            if ch == '"':
                assert value is not None
                value += ch
                state = BuildInfo._EnvParserState.VALUE
            else:
                raise ValueError("Improper formatting in Environment: couldn't interpret backslash sequence")
            continue
    if state != BuildInfo._EnvParserState.IGNORE_WHITESPACE:
        ValueError('Improper quoting in Environment: end quote not found')