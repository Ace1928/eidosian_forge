import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class LocationConfig(LockableConfig):
    """A configuration object that gives the policy for a location."""

    def __init__(self, location):
        super().__init__(file_name=bedding.locations_config_path())
        if location.startswith('file://'):
            location = urlutils.local_path_from_url(location)
        self.location = location

    def config_id(self):
        return 'locations'

    @classmethod
    def from_string(cls, str_or_unicode, location, save=False):
        """Create a config object from a string.

        Args:
          str_or_unicode: A string representing the file content. This will
            be utf-8 encoded.
          location: The location url to filter the configuration.
          save: Whether the file should be saved upon creation.
        """
        conf = cls(location)
        conf._create_from_string(str_or_unicode, save)
        return conf

    def _get_matching_sections(self):
        """Return an ordered list of section names matching this location."""
        matches = sorted(_iter_for_location_by_parts(self._get_parser(), self.location), key=lambda match: (match[2], match[0]), reverse=True)
        for section, extra_path, length in matches:
            yield (section, extra_path)
            try:
                if self._get_parser()[section].as_bool('ignore_parents'):
                    break
            except KeyError:
                pass

    def _get_sections(self, name=None):
        """See IniBasedConfig._get_sections()."""
        parser = self._get_parser()
        for name, extra_path in self._get_matching_sections():
            yield (name, parser[name], self.config_id())

    def _get_option_policy(self, section, option_name):
        """Return the policy for the given (section, option_name) pair."""
        try:
            recurse = self._get_parser()[section].as_bool('recurse')
        except KeyError:
            recurse = True
        if not recurse:
            return POLICY_NORECURSE
        policy_key = option_name + ':policy'
        try:
            policy_name = self._get_parser()[section][policy_key]
        except KeyError:
            policy_name = None
        return _policy_value[policy_name]

    def _set_option_policy(self, section, option_name, option_policy):
        """Set the policy for the given option name in the given section."""
        policy_key = option_name + ':policy'
        policy_name = _policy_name[option_policy]
        if policy_name is not None:
            self._get_parser()[section][policy_key] = policy_name
        elif policy_key in self._get_parser()[section]:
            del self._get_parser()[section][policy_key]

    def set_user_option(self, option, value, store=STORE_LOCATION):
        """Save option and its value in the configuration."""
        if store not in [STORE_LOCATION, STORE_LOCATION_NORECURSE, STORE_LOCATION_APPENDPATH]:
            raise ValueError('bad storage policy %r for %r' % (store, option))
        with self.lock_write():
            self.reload()
            location = self.location
            if location.endswith('/'):
                location = location[:-1]
            parser = self._get_parser()
            if location not in parser and (not location + '/' in parser):
                parser[location] = {}
            elif location + '/' in parser:
                location = location + '/'
            parser[location][option] = value
            self._set_option_policy(location, option, store)
            self._write_config_file()
            for hook in OldConfigHooks['set']:
                hook(self, option, value)