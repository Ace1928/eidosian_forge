from __future__ import annotations
import collections
import configparser
import copy
import os
import os.path
import re
from typing import (
from coverage.exceptions import ConfigError
from coverage.misc import isolate_module, human_sorted_items, substitute_variables
from coverage.tomlconfig import TomlConfigParser, TomlDecodeError
from coverage.types import (
class HandyConfigParser(configparser.ConfigParser):
    """Our specialization of ConfigParser."""

    def __init__(self, our_file: bool) -> None:
        """Create the HandyConfigParser.

        `our_file` is True if this config file is specifically for coverage,
        False if we are examining another config file (tox.ini, setup.cfg)
        for possible settings.
        """
        super().__init__(interpolation=None)
        self.section_prefixes = ['coverage:']
        if our_file:
            self.section_prefixes.append('')

    def read(self, filenames: Iterable[str], encoding_unused: str | None=None) -> list[str]:
        """Read a file name as UTF-8 configuration data."""
        return super().read(filenames, encoding='utf-8')

    def real_section(self, section: str) -> str | None:
        """Get the actual name of a section."""
        for section_prefix in self.section_prefixes:
            real_section = section_prefix + section
            has = super().has_section(real_section)
            if has:
                return real_section
        return None

    def has_option(self, section: str, option: str) -> bool:
        real_section = self.real_section(section)
        if real_section is not None:
            return super().has_option(real_section, option)
        return False

    def has_section(self, section: str) -> bool:
        return bool(self.real_section(section))

    def options(self, section: str) -> list[str]:
        real_section = self.real_section(section)
        if real_section is not None:
            return super().options(real_section)
        raise ConfigError(f'No section: {section!r}')

    def get_section(self, section: str) -> TConfigSectionOut:
        """Get the contents of a section, as a dictionary."""
        d: dict[str, TConfigValueOut] = {}
        for opt in self.options(section):
            d[opt] = self.get(section, opt)
        return d

    def get(self, section: str, option: str, *args: Any, **kwargs: Any) -> str:
        """Get a value, replacing environment variables also.

        The arguments are the same as `ConfigParser.get`, but in the found
        value, ``$WORD`` or ``${WORD}`` are replaced by the value of the
        environment variable ``WORD``.

        Returns the finished value.

        """
        for section_prefix in self.section_prefixes:
            real_section = section_prefix + section
            if super().has_option(real_section, option):
                break
        else:
            raise ConfigError(f'No option {option!r} in section: {section!r}')
        v: str = super().get(real_section, option, *args, **kwargs)
        v = substitute_variables(v, os.environ)
        return v

    def getlist(self, section: str, option: str) -> list[str]:
        """Read a list of strings.

        The value of `section` and `option` is treated as a comma- and newline-
        separated list of strings.  Each value is stripped of white space.

        Returns the list of strings.

        """
        value_list = self.get(section, option)
        values = []
        for value_line in value_list.split('\n'):
            for value in value_line.split(','):
                value = value.strip()
                if value:
                    values.append(value)
        return values

    def getregexlist(self, section: str, option: str) -> list[str]:
        """Read a list of full-line regexes.

        The value of `section` and `option` is treated as a newline-separated
        list of regexes.  Each value is stripped of white space.

        Returns the list of strings.

        """
        line_list = self.get(section, option)
        value_list = []
        for value in line_list.splitlines():
            value = value.strip()
            try:
                re.compile(value)
            except re.error as e:
                raise ConfigError(f'Invalid [{section}].{option} value {value!r}: {e}') from e
            if value:
                value_list.append(value)
        return value_list