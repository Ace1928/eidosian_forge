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
def _parse_requirements_file(self, requirements_file, allow_old_format=True, artifacts_manager=None, validate_signature_options=True):
    """
        Parses an Ansible requirement.yml file and returns all the roles and/or collections defined in it. There are 2
        requirements file format:

            # v1 (roles only)
            - src: The source of the role, required if include is not set. Can be Galaxy role name, URL to a SCM repo or tarball.
              name: Downloads the role to the specified name, defaults to Galaxy name from Galaxy or name of repo if src is a URL.
              scm: If src is a URL, specify the SCM. Only git or hd are supported and defaults ot git.
              version: The version of the role to download. Can also be tag, commit, or branch name and defaults to master.
              include: Path to additional requirements.yml files.

            # v2 (roles and collections)
            ---
            roles:
            # Same as v1 format just under the roles key

            collections:
            - namespace.collection
            - name: namespace.collection
              version: version identifier, multiple identifiers are separated by ','
              source: the URL or a predefined source name that relates to C.GALAXY_SERVER_LIST
              type: git|file|url|galaxy

        :param requirements_file: The path to the requirements file.
        :param allow_old_format: Will fail if a v1 requirements file is found and this is set to False.
        :param artifacts_manager: Artifacts manager.
        :return: a dict containing roles and collections to found in the requirements file.
        """
    requirements = {'roles': [], 'collections': []}
    b_requirements_file = to_bytes(requirements_file, errors='surrogate_or_strict')
    if not os.path.exists(b_requirements_file):
        raise AnsibleError("The requirements file '%s' does not exist." % to_native(requirements_file))
    display.vvv("Reading requirement file at '%s'" % requirements_file)
    with open(b_requirements_file, 'rb') as req_obj:
        try:
            file_requirements = yaml_load(req_obj)
        except YAMLError as err:
            raise AnsibleError("Failed to parse the requirements yml at '%s' with the following error:\n%s" % (to_native(requirements_file), to_native(err)))
    if file_requirements is None:
        raise AnsibleError("No requirements found in file '%s'" % to_native(requirements_file))

    def parse_role_req(requirement):
        if 'include' not in requirement:
            role = RoleRequirement.role_yaml_parse(requirement)
            display.vvv('found role %s in yaml file' % to_text(role))
            if 'name' not in role and 'src' not in role:
                raise AnsibleError('Must specify name or src for role')
            return [GalaxyRole(self.galaxy, self.lazy_role_api, **role)]
        else:
            b_include_path = to_bytes(requirement['include'], errors='surrogate_or_strict')
            if not os.path.isfile(b_include_path):
                raise AnsibleError("Failed to find include requirements file '%s' in '%s'" % (to_native(b_include_path), to_native(requirements_file)))
            with open(b_include_path, 'rb') as f_include:
                try:
                    return [GalaxyRole(self.galaxy, self.lazy_role_api, **r) for r in (RoleRequirement.role_yaml_parse(i) for i in yaml_load(f_include))]
                except Exception as e:
                    raise AnsibleError('Unable to load data from include requirements file: %s %s' % (to_native(requirements_file), to_native(e)))
    if isinstance(file_requirements, list):
        if not allow_old_format:
            raise AnsibleError("Expecting requirements file to be a dict with the key 'collections' that contains a list of collections to install")
        for role_req in file_requirements:
            requirements['roles'] += parse_role_req(role_req)
    elif isinstance(file_requirements, dict):
        extra_keys = set(file_requirements.keys()).difference(set(['roles', 'collections']))
        if extra_keys:
            raise AnsibleError("Expecting only 'roles' and/or 'collections' as base keys in the requirements file. Found: %s" % to_native(', '.join(extra_keys)))
        for role_req in file_requirements.get('roles') or []:
            requirements['roles'] += parse_role_req(role_req)
        requirements['collections'] = [Requirement.from_requirement_dict(self._init_coll_req_dict(collection_req), artifacts_manager, validate_signature_options) for collection_req in file_requirements.get('collections') or []]
    else:
        raise AnsibleError(f'Expecting requirements yaml to be a list or dictionary but got {type(file_requirements).__name__}')
    return requirements