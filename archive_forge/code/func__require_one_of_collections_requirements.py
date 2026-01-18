from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import argparse
import functools
import json
import os.path
import pathlib
import re
import shutil
import sys
import textwrap
import time
import typing as t
from dataclasses import dataclass
from yaml.error import YAMLError
import ansible.constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.galaxy import Galaxy, get_collections_galaxy_meta_info
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.galaxy.collection import (
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.gpg import GPG_ERROR_MAP
from ansible.galaxy.dependency_resolution.dataclasses import Requirement
from ansible.galaxy.role import GalaxyRole
from ansible.galaxy.token import BasicAuthToken, GalaxyToken, KeycloakToken, NoTokenSentinel
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils import six
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.playbook.role.requirement import RoleRequirement
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
def _require_one_of_collections_requirements(self, collections, requirements_file, signatures=None, artifacts_manager=None):
    if collections and requirements_file:
        raise AnsibleError('The positional collection_name arg and --requirements-file are mutually exclusive.')
    elif not collections and (not requirements_file):
        raise AnsibleError('You must specify a collection name or a requirements file.')
    elif requirements_file:
        if signatures is not None:
            raise AnsibleError("The --signatures option and --requirements-file are mutually exclusive. Use the --signatures with positional collection_name args or provide a 'signatures' key for requirements in the --requirements-file.")
        requirements_file = GalaxyCLI._resolve_path(requirements_file)
        requirements = self._parse_requirements_file(requirements_file, allow_old_format=False, artifacts_manager=artifacts_manager)
    else:
        requirements = {'collections': [Requirement.from_string(coll_input, artifacts_manager, signatures) for coll_input in collections], 'roles': []}
    return requirements