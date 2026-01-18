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
def _parse_cli_opts_from_config_file(self, config_file, sections, normalized):
    """Parse CLI options from a config file.

        CLI options are special - we require they be registered before the
        command line is parsed. This means that as we parse config files, we
        can go ahead and apply the appropriate option-type specific conversion
        to the values in config files for CLI options. We can't do this for
        non-CLI options, because the schema describing those options may not be
        registered until after the config files are parsed.

        This method relies on that invariant in order to enforce proper
        priority of option values - i.e. that the order in which an option
        value is parsed, whether the value comes from the CLI or a config file,
        determines which value specified for a given option wins.

        The way we implement this ordering is that as we parse each config
        file, we look for values in that config file for CLI options only. Any
        values for CLI options found in the config file are treated like they
        had appeared on the command line and set as attributes on the namespace
        objects. Values in later config files or on the command line will
        override values found in this file.
        """
    namespace = _Namespace(self._conf)
    namespace._add_parsed_config_file(config_file, sections, normalized)
    for opt, group in self._conf._all_cli_opts():
        group_name = group.name if group is not None else None
        try:
            value, loc = opt._get_from_namespace(namespace, group_name)
        except KeyError:
            continue
        except ValueError as ve:
            raise ConfigFileValueError('Value for option %s is not valid: %s' % (opt.name, str(ve)))
        if group_name is None:
            dest = opt.dest
        else:
            dest = group_name + '_' + opt.dest
        if opt.multi:
            if getattr(self, dest, None) is None:
                setattr(self, dest, [])
            values = getattr(self, dest)
            values.extend(value)
        else:
            setattr(self, dest, value)