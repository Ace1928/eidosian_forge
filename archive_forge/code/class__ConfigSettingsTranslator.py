import io
import os
import shlex
import sys
import tokenize
import shutil
import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import setuptools
import distutils
from . import errors
from ._path import same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning
from distutils.util import strtobool
class _ConfigSettingsTranslator:
    """Translate ``config_settings`` into distutils-style command arguments.
    Only a limited number of options is currently supported.
    """

    def _get_config(self, key: str, config_settings: _ConfigSettings) -> List[str]:
        """
        Get the value of a specific key in ``config_settings`` as a list of strings.

        >>> fn = _ConfigSettingsTranslator()._get_config
        >>> fn("--global-option", None)
        []
        >>> fn("--global-option", {})
        []
        >>> fn("--global-option", {'--global-option': 'foo'})
        ['foo']
        >>> fn("--global-option", {'--global-option': ['foo']})
        ['foo']
        >>> fn("--global-option", {'--global-option': 'foo'})
        ['foo']
        >>> fn("--global-option", {'--global-option': 'foo bar'})
        ['foo', 'bar']
        """
        cfg = config_settings or {}
        opts = cfg.get(key) or []
        return shlex.split(opts) if isinstance(opts, str) else opts

    def _global_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        Let the user specify ``verbose`` or ``quiet`` + escape hatch via
        ``--global-option``.
        Note: ``-v``, ``-vv``, ``-vvv`` have similar effects in setuptools,
        so we just have to cover the basic scenario ``-v``.

        >>> fn = _ConfigSettingsTranslator()._global_args
        >>> list(fn(None))
        []
        >>> list(fn({"verbose": "False"}))
        ['-q']
        >>> list(fn({"verbose": "1"}))
        ['-v']
        >>> list(fn({"--verbose": None}))
        ['-v']
        >>> list(fn({"verbose": "true", "--global-option": "-q --no-user-cfg"}))
        ['-v', '-q', '--no-user-cfg']
        >>> list(fn({"--quiet": None}))
        ['-q']
        """
        cfg = config_settings or {}
        falsey = {'false', 'no', '0', 'off'}
        if 'verbose' in cfg or '--verbose' in cfg:
            level = str(cfg.get('verbose') or cfg.get('--verbose') or '1')
            yield ('-q' if level.lower() in falsey else '-v')
        if 'quiet' in cfg or '--quiet' in cfg:
            level = str(cfg.get('quiet') or cfg.get('--quiet') or '1')
            yield ('-v' if level.lower() in falsey else '-q')
        yield from self._get_config('--global-option', config_settings)

    def __dist_info_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        The ``dist_info`` command accepts ``tag-date`` and ``tag-build``.

        .. warning::
           We cannot use this yet as it requires the ``sdist`` and ``bdist_wheel``
           commands run in ``build_sdist`` and ``build_wheel`` to reuse the egg-info
           directory created in ``prepare_metadata_for_build_wheel``.

        >>> fn = _ConfigSettingsTranslator()._ConfigSettingsTranslator__dist_info_args
        >>> list(fn(None))
        []
        >>> list(fn({"tag-date": "False"}))
        ['--no-date']
        >>> list(fn({"tag-date": None}))
        ['--no-date']
        >>> list(fn({"tag-date": "true", "tag-build": ".a"}))
        ['--tag-date', '--tag-build', '.a']
        """
        cfg = config_settings or {}
        if 'tag-date' in cfg:
            val = strtobool(str(cfg['tag-date'] or 'false'))
            yield ('--tag-date' if val else '--no-date')
        if 'tag-build' in cfg:
            yield from ['--tag-build', str(cfg['tag-build'])]

    def _editable_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        The ``editable_wheel`` command accepts ``editable-mode=strict``.

        >>> fn = _ConfigSettingsTranslator()._editable_args
        >>> list(fn(None))
        []
        >>> list(fn({"editable-mode": "strict"}))
        ['--mode', 'strict']
        """
        cfg = config_settings or {}
        mode = cfg.get('editable-mode') or cfg.get('editable_mode')
        if not mode:
            return
        yield from ['--mode', str(mode)]

    def _arbitrary_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        Users may expect to pass arbitrary lists of arguments to a command
        via "--global-option" (example provided in PEP 517 of a "escape hatch").

        >>> fn = _ConfigSettingsTranslator()._arbitrary_args
        >>> list(fn(None))
        []
        >>> list(fn({}))
        []
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': ['foo']}))
        ['foo']
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': 'foo bar'}))
        ['foo', 'bar']
        >>> list(fn({'--global-option': 'foo'}))
        []
        """
        yield from self._get_config('--build-option', config_settings)