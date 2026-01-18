from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def add_default_priv(self):
    for obj in self._objs:
        if self._as_who:
            self.query.append('ALTER DEFAULT PRIVILEGES FOR ROLE {0}{1} GRANT {2} ON {3} TO {4}'.format(self._as_who, self._schema, self._set_what, obj, self._for_whom))
        else:
            self.query.append('ALTER DEFAULT PRIVILEGES{0} GRANT {1} ON {2} TO {3}'.format(self._schema, self._set_what, obj, self._for_whom))
        self.add_grant_option()