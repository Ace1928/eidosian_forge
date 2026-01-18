import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
class SubCommandOpt(Opt):
    """Sub-command options.

    Sub-command options allow argparse sub-parsers to be used to parse
    additional command line arguments.

    The handler argument to the SubCommandOpt constructor is a callable
    which is supplied an argparse subparsers object. Use this handler
    callable to add sub-parsers.

    The opt value is SubCommandAttr object with the name of the chosen
    sub-parser stored in the 'name' attribute and the values of other
    sub-parser arguments available as additional attributes.

    :param name: the option's name
    :param dest: the name of the corresponding :class:`.ConfigOpts` property
    :param handler: callable which is supplied subparsers object when invoked
    :param title: title of the sub-commands group in help output
    :param description: description of the group in help output
    :param help: a help string giving an overview of available sub-commands
    """

    def __init__(self, name, dest=None, handler=None, title=None, description=None, help=None):
        """Construct an sub-command parsing option.

        This behaves similarly to other Opt sub-classes but adds a
        'handler' argument. The handler is a callable which is supplied
        an subparsers object when invoked. The add_parser() method on
        this subparsers object can be used to register parsers for
        sub-commands.
        """
        super(SubCommandOpt, self).__init__(name, type=types.String(), dest=dest, help=help)
        self.handler = handler
        self.title = title
        self.description = description

    def _add_to_cli(self, parser, group=None):
        """Add argparse sub-parsers and invoke the handler method."""
        dest = self.dest
        if group is not None:
            dest = group.name + '_' + dest
        subparsers = parser.add_subparsers(dest=dest, title=self.title, description=self.description, help=self.help)
        subparsers.required = True
        if self.handler is not None:
            self.handler(subparsers)