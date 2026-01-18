from __future__ import annotations
import argparse
import enum
import functools
import logging
from typing import Any
from typing import Callable
from typing import Sequence
from flake8 import utils
from flake8.plugins.finder import Plugins
class OptionManager:
    """Manage Options and OptionParser while adding post-processing."""

    def __init__(self, *, version: str, plugin_versions: str, parents: list[argparse.ArgumentParser], formatter_names: list[str]) -> None:
        """Initialize an instance of an OptionManager."""
        self.formatter_names = formatter_names
        self.parser = argparse.ArgumentParser(prog='flake8', usage='%(prog)s [options] file file ...', parents=parents, epilog=f'Installed plugins: {plugin_versions}')
        self.parser.add_argument('--version', action='version', version=f'{version} ({plugin_versions}) {utils.get_python_version()}')
        self.parser.add_argument('filenames', nargs='*', metavar='filename')
        self.config_options_dict: dict[str, Option] = {}
        self.options: list[Option] = []
        self.extended_default_ignore: list[str] = []
        self.extended_default_select: list[str] = []
        self._current_group: argparse._ArgumentGroup | None = None

    def register_plugins(self, plugins: Plugins) -> None:
        """Register the plugin options (if needed)."""
        groups: dict[str, argparse._ArgumentGroup] = {}

        def _set_group(name: str) -> None:
            try:
                self._current_group = groups[name]
            except KeyError:
                group = self.parser.add_argument_group(name)
                self._current_group = groups[name] = group
        for loaded in plugins.all_plugins():
            add_options = getattr(loaded.obj, 'add_options', None)
            if add_options:
                _set_group(loaded.plugin.package)
                add_options(self)
            if loaded.plugin.entry_point.group == 'flake8.extension':
                self.extend_default_select([loaded.entry_name])
        self._current_group = None

    def add_option(self, *args: Any, **kwargs: Any) -> None:
        """Create and register a new option.

        See parameters for :class:`~flake8.options.manager.Option` for
        acceptable arguments to this method.

        .. note::

            ``short_option_name`` and ``long_option_name`` may be specified
            positionally as they are with argparse normally.
        """
        option = Option(*args, **kwargs)
        option_args, option_kwargs = option.to_argparse()
        if self._current_group is not None:
            self._current_group.add_argument(*option_args, **option_kwargs)
        else:
            self.parser.add_argument(*option_args, **option_kwargs)
        self.options.append(option)
        if option.parse_from_config:
            name = option.config_name
            assert name is not None
            self.config_options_dict[name] = option
            self.config_options_dict[name.replace('_', '-')] = option
        LOG.debug('Registered option "%s".', option)

    def extend_default_ignore(self, error_codes: Sequence[str]) -> None:
        """Extend the default ignore list with the error codes provided.

        :param error_codes:
            List of strings that are the error/warning codes with which to
            extend the default ignore list.
        """
        LOG.debug('Extending default ignore list with %r', error_codes)
        self.extended_default_ignore.extend(error_codes)

    def extend_default_select(self, error_codes: Sequence[str]) -> None:
        """Extend the default select list with the error codes provided.

        :param error_codes:
            List of strings that are the error/warning codes with which
            to extend the default select list.
        """
        LOG.debug('Extending default select list with %r', error_codes)
        self.extended_default_select.extend(error_codes)

    def parse_args(self, args: Sequence[str] | None=None, values: argparse.Namespace | None=None) -> argparse.Namespace:
        """Proxy to calling the OptionParser's parse_args method."""
        if values:
            self.parser.set_defaults(**vars(values))
        return self.parser.parse_args(args)