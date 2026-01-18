import configparser
from os import path
from typing import Dict, Optional
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
class UserTheme(Theme):
    """A user defined LaTeX theme."""
    REQUIRED_CONFIG_KEYS = ['docclass', 'wrapperclass']
    OPTIONAL_CONFIG_KEYS = ['papersize', 'pointsize', 'toplevel_sectioning']

    def __init__(self, name: str, filename: str) -> None:
        super().__init__(name)
        self.config = configparser.RawConfigParser()
        self.config.read(path.join(filename), encoding='utf-8')
        for key in self.REQUIRED_CONFIG_KEYS:
            try:
                value = self.config.get('theme', key)
                setattr(self, key, value)
            except configparser.NoSectionError as exc:
                raise ThemeError(__('%r doesn\'t have "theme" setting') % filename) from exc
            except configparser.NoOptionError as exc:
                raise ThemeError(__('%r doesn\'t have "%s" setting') % (filename, exc.args[0])) from exc
        for key in self.OPTIONAL_CONFIG_KEYS:
            try:
                value = self.config.get('theme', key)
                setattr(self, key, value)
            except configparser.NoOptionError:
                pass