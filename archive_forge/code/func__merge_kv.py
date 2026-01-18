from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.parsing.mod_args import ModuleArgsParser
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject, AnsibleMapping
from ansible.plugins.loader import lookup_loader
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.block import Block
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.loop_control import LoopControl
from ansible.playbook.notifiable import Notifiable
from ansible.playbook.role import Role
from ansible.playbook.taggable import Taggable
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
def _merge_kv(self, ds):
    if ds is None:
        return ''
    elif isinstance(ds, string_types):
        return ds
    elif isinstance(ds, dict):
        buf = ''
        for k, v in ds.items():
            if k.startswith('_'):
                continue
            buf = buf + '%s=%s ' % (k, v)
        buf = buf.strip()
        return buf