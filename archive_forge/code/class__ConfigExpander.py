import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Set, Union
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic
from ._apply_pyprojecttoml import apply as _apply
class _ConfigExpander:

    def __init__(self, config: dict, root_dir: Optional[_Path]=None, ignore_option_errors: bool=False, dist: Optional['Distribution']=None):
        self.config = config
        self.root_dir = root_dir or os.getcwd()
        self.project_cfg = config.get('project', {})
        self.dynamic = self.project_cfg.get('dynamic', [])
        self.setuptools_cfg = config.get('tool', {}).get('setuptools', {})
        self.dynamic_cfg = self.setuptools_cfg.get('dynamic', {})
        self.ignore_option_errors = ignore_option_errors
        self._dist = dist
        self._referenced_files: Set[str] = set()

    def _ensure_dist(self) -> 'Distribution':
        from setuptools.dist import Distribution
        attrs = {'src_root': self.root_dir, 'name': self.project_cfg.get('name', None)}
        return self._dist or Distribution(attrs)

    def _process_field(self, container: dict, field: str, fn: Callable):
        if field in container:
            with _ignore_errors(self.ignore_option_errors):
                container[field] = fn(container[field])

    def _canonic_package_data(self, field='package-data'):
        package_data = self.setuptools_cfg.get(field, {})
        return _expand.canonic_package_data(package_data)

    def expand(self):
        self._expand_packages()
        self._canonic_package_data()
        self._canonic_package_data('exclude-package-data')
        dist = self._ensure_dist()
        ctx = _EnsurePackagesDiscovered(dist, self.project_cfg, self.setuptools_cfg)
        with ctx as ensure_discovered:
            package_dir = ensure_discovered.package_dir
            self._expand_data_files()
            self._expand_cmdclass(package_dir)
            self._expand_all_dynamic(dist, package_dir)
        dist._referenced_files.update(self._referenced_files)
        return self.config

    def _expand_packages(self):
        packages = self.setuptools_cfg.get('packages')
        if packages is None or isinstance(packages, (list, tuple)):
            return
        find = packages.get('find')
        if isinstance(find, dict):
            find['root_dir'] = self.root_dir
            find['fill_package_dir'] = self.setuptools_cfg.setdefault('package-dir', {})
            with _ignore_errors(self.ignore_option_errors):
                self.setuptools_cfg['packages'] = _expand.find_packages(**find)

    def _expand_data_files(self):
        data_files = partial(_expand.canonic_data_files, root_dir=self.root_dir)
        self._process_field(self.setuptools_cfg, 'data-files', data_files)

    def _expand_cmdclass(self, package_dir: Mapping[str, str]):
        root_dir = self.root_dir
        cmdclass = partial(_expand.cmdclass, package_dir=package_dir, root_dir=root_dir)
        self._process_field(self.setuptools_cfg, 'cmdclass', cmdclass)

    def _expand_all_dynamic(self, dist: 'Distribution', package_dir: Mapping[str, str]):
        special = ('version', 'readme', 'entry-points', 'scripts', 'gui-scripts', 'classifiers', 'dependencies', 'optional-dependencies')
        obtained_dynamic = {field: self._obtain(dist, field, package_dir) for field in self.dynamic if field not in special}
        obtained_dynamic.update(self._obtain_entry_points(dist, package_dir) or {}, version=self._obtain_version(dist, package_dir), readme=self._obtain_readme(dist), classifiers=self._obtain_classifiers(dist), dependencies=self._obtain_dependencies(dist), optional_dependencies=self._obtain_optional_dependencies(dist))
        updates = {k: v for k, v in obtained_dynamic.items() if v is not None}
        self.project_cfg.update(updates)

    def _ensure_previously_set(self, dist: 'Distribution', field: str):
        previous = _PREVIOUSLY_DEFINED[field](dist)
        if previous is None and (not self.ignore_option_errors):
            msg = f'No configuration found for dynamic {field!r}.\nSome dynamic fields need to be specified via `tool.setuptools.dynamic`\nothers must be specified via the equivalent attribute in `setup.py`.'
            raise InvalidConfigError(msg)

    def _expand_directive(self, specifier: str, directive, package_dir: Mapping[str, str]):
        from setuptools.extern.more_itertools import always_iterable
        with _ignore_errors(self.ignore_option_errors):
            root_dir = self.root_dir
            if 'file' in directive:
                self._referenced_files.update(always_iterable(directive['file']))
                return _expand.read_files(directive['file'], root_dir)
            if 'attr' in directive:
                return _expand.read_attr(directive['attr'], package_dir, root_dir)
            raise ValueError(f'invalid `{specifier}`: {directive!r}')
        return None

    def _obtain(self, dist: 'Distribution', field: str, package_dir: Mapping[str, str]):
        if field in self.dynamic_cfg:
            return self._expand_directive(f'tool.setuptools.dynamic.{field}', self.dynamic_cfg[field], package_dir)
        self._ensure_previously_set(dist, field)
        return None

    def _obtain_version(self, dist: 'Distribution', package_dir: Mapping[str, str]):
        if 'version' in self.dynamic and 'version' in self.dynamic_cfg:
            return _expand.version(self._obtain(dist, 'version', package_dir))
        return None

    def _obtain_readme(self, dist: 'Distribution') -> Optional[Dict[str, str]]:
        if 'readme' not in self.dynamic:
            return None
        dynamic_cfg = self.dynamic_cfg
        if 'readme' in dynamic_cfg:
            return {'text': self._obtain(dist, 'readme', {}), 'content-type': dynamic_cfg['readme'].get('content-type', 'text/x-rst')}
        self._ensure_previously_set(dist, 'readme')
        return None

    def _obtain_entry_points(self, dist: 'Distribution', package_dir: Mapping[str, str]) -> Optional[Dict[str, dict]]:
        fields = ('entry-points', 'scripts', 'gui-scripts')
        if not any((field in self.dynamic for field in fields)):
            return None
        text = self._obtain(dist, 'entry-points', package_dir)
        if text is None:
            return None
        groups = _expand.entry_points(text)
        expanded = {'entry-points': groups}

        def _set_scripts(field: str, group: str):
            if group in groups:
                value = groups.pop(group)
                if field not in self.dynamic:
                    raise InvalidConfigError(_MissingDynamic.details(field, value))
                expanded[field] = value
        _set_scripts('scripts', 'console_scripts')
        _set_scripts('gui-scripts', 'gui_scripts')
        return expanded

    def _obtain_classifiers(self, dist: 'Distribution'):
        if 'classifiers' in self.dynamic:
            value = self._obtain(dist, 'classifiers', {})
            if value:
                return value.splitlines()
        return None

    def _obtain_dependencies(self, dist: 'Distribution'):
        if 'dependencies' in self.dynamic:
            value = self._obtain(dist, 'dependencies', {})
            if value:
                return _parse_requirements_list(value)
        return None

    def _obtain_optional_dependencies(self, dist: 'Distribution'):
        if 'optional-dependencies' not in self.dynamic:
            return None
        if 'optional-dependencies' in self.dynamic_cfg:
            optional_dependencies_map = self.dynamic_cfg['optional-dependencies']
            assert isinstance(optional_dependencies_map, dict)
            return {group: _parse_requirements_list(self._expand_directive(f'tool.setuptools.dynamic.optional-dependencies.{group}', directive, {})) for group, directive in optional_dependencies_map.items()}
        self._ensure_previously_set(dist, 'optional-dependencies')
        return None