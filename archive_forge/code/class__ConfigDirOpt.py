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
class _ConfigDirOpt(Opt):
    """The --config-dir option.

    This is an private option type which handles the special processing
    required for --config-dir options.

    As each --config-dir option is encountered on the command line, we
    parse the files in that directory and store the parsed values in the
    _Namespace object. This allows us to properly handle the precedence of
    --config-dir options over previous command line arguments, but not
    over subsequent arguments.

    .. versionadded:: 1.2
    """

    class ConfigDirAction(argparse.Action):
        """An argparse action for --config-dir.

        As each --config-dir option is encountered, this action sets the
        config_dir attribute on the _Namespace object but also parses the
        configuration files and stores the values found also in the
        _Namespace object.
        """

        def __call__(self, parser, namespace, values, option_string=None):
            """Handle a --config-dir command line argument.

            :raises: ConfigFileParseError, ConfigFileValueError,
                     ConfigDirNotFoundError
            """
            namespace._config_dirs.append(values)
            setattr(namespace, self.dest, values)
            values = os.path.expanduser(values)
            if not os.path.exists(values):
                raise ConfigDirNotFoundError(values)
            config_dir_glob = os.path.join(values, '*.conf')
            for config_file in sorted(glob.glob(config_dir_glob)):
                ConfigParser._parse_file(config_file, namespace)

    def __init__(self, name, **kwargs):
        super(_ConfigDirOpt, self).__init__(name, type=types.List(), **kwargs)

    def _get_argparse_kwargs(self, group, **kwargs):
        """Extends the base argparse keyword dict for the config dir option."""
        kwargs = super(_ConfigDirOpt, self)._get_argparse_kwargs(group)
        kwargs['action'] = self.ConfigDirAction
        return kwargs