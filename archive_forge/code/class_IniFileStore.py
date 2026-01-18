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
class IniFileStore(Store):
    """A config Store using ConfigObj for storage.

    :ivar _config_obj: Private member to hold the ConfigObj instance used to
        serialize/deserialize the config file.
    """

    def __init__(self):
        """A config Store using ConfigObj for storage.
        """
        super().__init__()
        self._config_obj = None

    def is_loaded(self):
        return self._config_obj is not None

    def unload(self):
        self._config_obj = None
        self.dirty_sections = {}

    def _load_content(self):
        """Load the config file bytes.

        This should be provided by subclasses

        Returns:
          Byte string
        """
        raise NotImplementedError(self._load_content)

    def _save_content(self, content):
        """Save the config file bytes.

        This should be provided by subclasses

        Args:
          content: Config file bytes to write
        """
        raise NotImplementedError(self._save_content)

    def load(self):
        """Load the store from the associated file."""
        if self.is_loaded():
            return
        content = self._load_content()
        self._load_from_string(content)
        for hook in ConfigHooks['load']:
            hook(self)

    def _load_from_string(self, bytes):
        """Create a config store from a string.

        Args:
          bytes: A string representing the file content.
        """
        if self.is_loaded():
            raise AssertionError('Already loaded: {!r}'.format(self._config_obj))
        co_input = BytesIO(bytes)
        try:
            self._config_obj = ConfigObj(co_input, encoding='utf-8', list_values=False)
        except configobj.ConfigObjError as e:
            self._config_obj = None
            raise ParseConfigError(e.errors, self.external_url())
        except UnicodeDecodeError:
            raise ConfigContentError(self.external_url())

    def save_changes(self):
        if not self.is_loaded():
            return
        if not self._need_saving():
            return
        dirty_sections = self.dirty_sections.copy()
        self.apply_changes(dirty_sections)
        self.save()

    def save(self):
        if not self.is_loaded():
            return
        out = BytesIO()
        self._config_obj.write(out)
        self._save_content(out.getvalue())
        for hook in ConfigHooks['save']:
            hook(self)

    def get_sections(self) -> Iterable[Tuple[Store, Section]]:
        """Get the configobj section in the file order.

        Returns: An iterable of (store, section).
        """
        try:
            self.load()
        except (transport.NoSuchFile, errors.PermissionDenied):
            return
        cobj = self._config_obj
        if cobj.scalars:
            yield (self, self.readonly_section_class(None, cobj))
        for section_name in cobj.sections:
            yield (self, self.readonly_section_class(section_name, cobj[section_name]))

    def get_mutable_section(self, section_id=None):
        try:
            self.load()
        except transport.NoSuchFile:
            self._load_from_string(b'')
        if section_id in self.dirty_sections:
            return self.dirty_sections[section_id]
        if section_id is None:
            section = self._config_obj
        else:
            section = self._config_obj.setdefault(section_id, {})
        mutable_section = self.mutable_section_class(section_id, section)
        self.dirty_sections[section_id] = mutable_section
        return mutable_section

    def quote(self, value):
        try:
            self._config_obj.list_values = True
            return self._config_obj._quote(value)
        finally:
            self._config_obj.list_values = False

    def unquote(self, value):
        if value and isinstance(value, str):
            value = self._config_obj._unquote(value)
        return value

    def external_url(self):
        return 'In-Process Store, no URL'