import contextlib
import functools
import os
from collections import defaultdict
from functools import partial
from functools import wraps
from typing import (
from ..errors import FileError, OptionError
from ..extern.packaging.markers import default_environment as marker_env
from ..extern.packaging.requirements import InvalidRequirement, Requirement
from ..extern.packaging.specifiers import SpecifierSet
from ..extern.packaging.version import InvalidVersion, Version
from ..warnings import SetuptoolsDeprecationWarning
from . import expand
class ConfigOptionsHandler(ConfigHandler['Distribution']):
    section_prefix = 'options'

    def __init__(self, target_obj: 'Distribution', options: AllCommandOptions, ignore_option_errors: bool, ensure_discovered: expand.EnsurePackagesDiscovered):
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.root_dir = target_obj.src_root
        self.package_dir: Dict[str, str] = {}

    @classmethod
    def _parse_list_semicolon(cls, value):
        return cls._parse_list(value, separator=';')

    def _parse_file_in_root(self, value):
        return self._parse_file(value, root_dir=self.root_dir)

    def _parse_requirements_list(self, label: str, value: str):
        parsed = self._parse_list_semicolon(self._parse_file_in_root(value))
        _warn_accidental_env_marker_misconfig(label, value, parsed)
        return [line for line in parsed if not line.startswith('#')]

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_bool = self._parse_bool
        parse_dict = self._parse_dict
        parse_cmdclass = self._parse_cmdclass
        return {'zip_safe': parse_bool, 'include_package_data': parse_bool, 'package_dir': parse_dict, 'scripts': parse_list, 'eager_resources': parse_list, 'dependency_links': parse_list, 'namespace_packages': self._deprecated_config_handler(parse_list, 'The namespace_packages parameter is deprecated, consider using implicit namespaces instead (PEP 420).'), 'install_requires': partial(self._parse_requirements_list, 'install_requires'), 'setup_requires': self._parse_list_semicolon, 'tests_require': self._parse_list_semicolon, 'packages': self._parse_packages, 'entry_points': self._parse_file_in_root, 'py_modules': parse_list, 'python_requires': SpecifierSet, 'cmdclass': parse_cmdclass}

    def _parse_cmdclass(self, value):
        package_dir = self.ensure_discovered.package_dir
        return expand.cmdclass(self._parse_dict(value), package_dir, self.root_dir)

    def _parse_packages(self, value):
        """Parses `packages` option value.

        :param value:
        :rtype: list
        """
        find_directives = ['find:', 'find_namespace:']
        trimmed_value = value.strip()
        if trimmed_value not in find_directives:
            return self._parse_list(value)
        find_kwargs = self.parse_section_packages__find(self.sections.get('packages.find', {}))
        find_kwargs.update(namespaces=trimmed_value == find_directives[1], root_dir=self.root_dir, fill_package_dir=self.package_dir)
        return expand.find_packages(**find_kwargs)

    def parse_section_packages__find(self, section_options):
        """Parses `packages.find` configuration file section.

        To be used in conjunction with _parse_packages().

        :param dict section_options:
        """
        section_data = self._parse_section_to_dict(section_options, self._parse_list)
        valid_keys = ['where', 'include', 'exclude']
        find_kwargs = dict([(k, v) for k, v in section_data.items() if k in valid_keys and v])
        where = find_kwargs.get('where')
        if where is not None:
            find_kwargs['where'] = where[0]
        return find_kwargs

    def parse_section_entry_points(self, section_options):
        """Parses `entry_points` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['entry_points'] = parsed

    def _parse_package_data(self, section_options):
        package_data = self._parse_section_to_dict(section_options, self._parse_list)
        return expand.canonic_package_data(package_data)

    def parse_section_package_data(self, section_options):
        """Parses `package_data` configuration file section.

        :param dict section_options:
        """
        self['package_data'] = self._parse_package_data(section_options)

    def parse_section_exclude_package_data(self, section_options):
        """Parses `exclude_package_data` configuration file section.

        :param dict section_options:
        """
        self['exclude_package_data'] = self._parse_package_data(section_options)

    def parse_section_extras_require(self, section_options):
        """Parses `extras_require` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict_with_key(section_options, lambda k, v: self._parse_requirements_list(f'extras_require[{k}]', v))
        self['extras_require'] = parsed

    def parse_section_data_files(self, section_options):
        """Parses `data_files` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['data_files'] = expand.canonic_data_files(parsed, self.root_dir)