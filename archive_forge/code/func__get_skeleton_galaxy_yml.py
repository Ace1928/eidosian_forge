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
@staticmethod
def _get_skeleton_galaxy_yml(template_path, inject_data):
    with open(to_bytes(template_path, errors='surrogate_or_strict'), 'rb') as template_obj:
        meta_template = to_text(template_obj.read(), errors='surrogate_or_strict')
    galaxy_meta = get_collections_galaxy_meta_info()
    required_config = []
    optional_config = []
    for meta_entry in galaxy_meta:
        config_list = required_config if meta_entry.get('required', False) else optional_config
        value = inject_data.get(meta_entry['key'], None)
        if not value:
            meta_type = meta_entry.get('type', 'str')
            if meta_type == 'str':
                value = ''
            elif meta_type == 'list':
                value = []
            elif meta_type == 'dict':
                value = {}
        meta_entry['value'] = value
        config_list.append(meta_entry)
    link_pattern = re.compile('L\\(([^)]+),\\s+([^)]+)\\)')
    const_pattern = re.compile('C\\(([^)]+)\\)')

    def comment_ify(v):
        if isinstance(v, list):
            v = '. '.join([l.rstrip('.') for l in v])
        v = link_pattern.sub('\\1 <\\2>', v)
        v = const_pattern.sub("'\\1'", v)
        return textwrap.fill(v, width=117, initial_indent='# ', subsequent_indent='# ', break_on_hyphens=False)
    loader = DataLoader()
    templar = Templar(loader, variables={'required_config': required_config, 'optional_config': optional_config})
    templar.environment.filters['comment_ify'] = comment_ify
    meta_value = templar.template(meta_template)
    return meta_value