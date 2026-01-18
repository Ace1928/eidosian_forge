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
class ConfigMetadataHandler(ConfigHandler['DistributionMetadata']):
    section_prefix = 'metadata'
    aliases = {'home_page': 'url', 'summary': 'description', 'classifier': 'classifiers', 'platform': 'platforms'}
    strict_mode = False
    'We need to keep it loose, to be partially compatible with\n    `pbr` and `d2to1` packages which also uses `metadata` section.\n\n    '

    def __init__(self, target_obj: 'DistributionMetadata', options: AllCommandOptions, ignore_option_errors: bool, ensure_discovered: expand.EnsurePackagesDiscovered, package_dir: Optional[dict]=None, root_dir: _Path=os.curdir):
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.package_dir = package_dir
        self.root_dir = root_dir

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_file = partial(self._parse_file, root_dir=self.root_dir)
        parse_dict = self._parse_dict
        exclude_files_parser = self._exclude_files_parser
        return {'platforms': parse_list, 'keywords': parse_list, 'provides': parse_list, 'obsoletes': parse_list, 'classifiers': self._get_parser_compound(parse_file, parse_list), 'license': exclude_files_parser('license'), 'license_files': parse_list, 'description': parse_file, 'long_description': parse_file, 'version': self._parse_version, 'project_urls': parse_dict}

    def _parse_version(self, value):
        """Parses `version` option value.

        :param value:
        :rtype: str

        """
        version = self._parse_file(value, self.root_dir)
        if version != value:
            version = version.strip()
            try:
                Version(version)
            except InvalidVersion as e:
                raise OptionError(f'Version loaded from {value} does not comply with PEP 440: {version}') from e
            return version
        return expand.version(self._parse_attr(value, self.package_dir, self.root_dir))