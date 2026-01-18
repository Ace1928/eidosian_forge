from __future__ import (absolute_import, division, print_function)
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from hashlib import sha1
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleFileNotFound, AnsibleAssertionError, AnsibleTemplateError
from ansible.inventory.host import Host
from ansible.inventory.helpers import sort_groups, get_group_vars
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import text_type, string_types
from ansible.plugins.loader import lookup_loader
from ansible.vars.fact_cache import FactCache
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.vars import combine_vars, load_extra_vars, load_options_vars
from ansible.utils.unsafe_proxy import wrap_var
from ansible.vars.clean import namespace_facts, clean_facts
from ansible.vars.plugins import get_vars_from_inventory_sources, get_vars_from_path
def _get_magic_variables(self, play, host, task, include_hostvars, _hosts=None, _hosts_all=None):
    """
        Returns a dictionary of so-called "magic" variables in Ansible,
        which are special variables we set internally for use.
        """
    variables = {}
    variables['playbook_dir'] = os.path.abspath(self._loader.get_basedir())
    variables['ansible_playbook_python'] = sys.executable
    variables['ansible_config_file'] = C.CONFIG_FILE
    if play:
        dependency_role_names = list({d.get_name() for r in play.roles for d in r.get_all_dependencies()})
        play_role_names = [r.get_name() for r in play.roles]
        variables['ansible_role_names'] = list(set(dependency_role_names + play_role_names))
        variables['ansible_play_role_names'] = play_role_names
        variables['ansible_dependent_role_names'] = dependency_role_names
        variables['role_names'] = variables['ansible_play_role_names']
        variables['ansible_play_name'] = play.get_name()
    if task:
        if task._role:
            variables['role_name'] = task._role.get_name(include_role_fqcn=False)
            variables['role_path'] = task._role._role_path
            variables['role_uuid'] = text_type(task._role._uuid)
            variables['ansible_collection_name'] = task._role._role_collection
            variables['ansible_role_name'] = task._role.get_name()
    if self._inventory is not None:
        variables['groups'] = self._inventory.get_groups_dict()
        if play:
            templar = Templar(loader=self._loader)
            if not play.finalized and templar.is_template(play.hosts):
                pattern = 'all'
            else:
                pattern = play.hosts or 'all'
            if not _hosts_all:
                _hosts_all = [h.name for h in self._inventory.get_hosts(pattern=pattern, ignore_restrictions=True)]
            if not _hosts:
                _hosts = [h.name for h in self._inventory.get_hosts()]
            variables['ansible_play_hosts_all'] = _hosts_all[:]
            variables['ansible_play_hosts'] = [x for x in variables['ansible_play_hosts_all'] if x not in play._removed_hosts]
            variables['ansible_play_batch'] = [x for x in _hosts if x not in play._removed_hosts]
            variables['play_hosts'] = variables['ansible_play_batch']
    variables['omit'] = self._omit_token
    for option, option_value in self._options_vars.items():
        variables[option] = option_value
    if self._hostvars is not None and include_hostvars:
        variables['hostvars'] = self._hostvars
    return variables