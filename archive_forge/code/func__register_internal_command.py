import click
import os
import shlex
import sys
from collections import defaultdict
from .exceptions import CommandLineParserError, ExitReplException
def _register_internal_command(names, target, description=None):
    if not hasattr(target, '__call__'):
        raise ValueError('Internal command must be a callable')
    if isinstance(names, str):
        names = [names]
    elif isinstance(names, Mapping) or not isinstance(names, Iterable):
        raise ValueError('"names" must be a string, or an iterable object, but got "{}"'.format(type(names).__name__))
    for name in names:
        _internal_commands[name] = (target, description)