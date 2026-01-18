from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import stat
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError
from ansible.executor.playbook_executor import PlaybookExecutor
from ansible.module_utils.common.text.converters import to_bytes
from ansible.playbook.block import Block
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.utils.display import Display
@staticmethod
def _flush_cache(inventory, variable_manager):
    for host in inventory.list_hosts():
        hostname = host.get_name()
        variable_manager.clear_facts(hostname)