from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class ControlComponentFormatRegistry(registry.FormatRegistry[ControlComponentFormat]):
    """A registry for control components (branch, workingtree, repository)."""

    def __init__(self, other_registry=None):
        super().__init__(other_registry)
        self._extra_formats = []

    def register(self, format):
        """Register a new format."""
        super().register(format.get_format_string(), format)

    def remove(self, format):
        """Remove a registered format."""
        super().remove(format.get_format_string())

    def register_extra(self, format):
        """Register a format that can not be used in a metadir.

        This is mainly useful to allow custom repository formats, such as older
        Bazaar formats and foreign formats, to be tested.
        """
        self._extra_formats.append(registry._ObjectGetter(format))

    def remove_extra(self, format):
        """Remove an extra format.
        """
        self._extra_formats.remove(registry._ObjectGetter(format))

    def register_extra_lazy(self, module_name, member_name):
        """Register a format lazily.
        """
        self._extra_formats.append(registry._LazyObjectGetter(module_name, member_name))

    def _get_extra(self):
        """Return getters for extra formats, not usable in meta directories."""
        return [getter.get_obj for getter in self._extra_formats]

    def _get_all_lazy(self):
        """Return getters for all formats, even those not usable in metadirs."""
        result = [self._dict[name].get_obj for name in self.keys()]
        result.extend(self._get_extra())
        return result

    def _get_all(self):
        """Return all formats, even those not usable in metadirs."""
        result = []
        for getter in self._get_all_lazy():
            fmt = getter()
            if callable(fmt):
                fmt = fmt()
            result.append(fmt)
        return result

    def _get_all_modules(self):
        """Return a set of the modules providing objects."""
        modules = set()
        for name in self.keys():
            modules.add(self._get_module(name))
        for getter in self._extra_formats:
            modules.add(getter.get_module())
        return modules