from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
class QueryBuilder(object):

    def __init__(self, state):
        self._grant_option = None
        self._for_whom = None
        self._as_who = None
        self._set_what = None
        self._obj_type = None
        self._state = state
        self._schema = None
        self._objs = None
        self.query = []

    def for_objs(self, objs):
        self._objs = objs
        return self

    def for_schema(self, schema):
        self._schema = ' IN SCHEMA %s' % schema if schema is not None else ''
        return self

    def with_grant_option(self, option):
        self._grant_option = option
        return self

    def for_whom(self, who):
        self._for_whom = who
        return self

    def as_who(self, target_roles):
        self._as_who = target_roles
        return self

    def set_what(self, what):
        self._set_what = what
        return self

    def for_objtype(self, objtype):
        self._obj_type = objtype
        return self

    def build(self):
        if self._state == 'present':
            self.build_present()
        elif self._state == 'absent':
            self.build_absent()
        else:
            self.build_absent()
        return '\n'.join(self.query)

    def add_default_revoke(self):
        for obj in self._objs:
            if self._as_who:
                self.query.append('ALTER DEFAULT PRIVILEGES FOR ROLE {0}{1} REVOKE ALL ON {2} FROM {3};'.format(self._as_who, self._schema, obj, self._for_whom))
            else:
                self.query.append('ALTER DEFAULT PRIVILEGES{0} REVOKE ALL ON {1} FROM {2};'.format(self._schema, obj, self._for_whom))

    def add_grant_option(self):
        if self._grant_option:
            if self._obj_type == 'group':
                self.query[-1] += ' WITH ADMIN OPTION;'
            else:
                self.query[-1] += ' WITH GRANT OPTION;'
        elif self._grant_option is False:
            self.query[-1] += ';'
            if self._obj_type == 'group':
                self.query.append('REVOKE ADMIN OPTION FOR {0} FROM {1};'.format(self._set_what, self._for_whom))
            elif not self._obj_type == 'default_privs':
                self.query.append('REVOKE GRANT OPTION FOR {0} FROM {1};'.format(self._set_what, self._for_whom))
        else:
            self.query[-1] += ';'

    def add_default_priv(self):
        for obj in self._objs:
            if self._as_who:
                self.query.append('ALTER DEFAULT PRIVILEGES FOR ROLE {0}{1} GRANT {2} ON {3} TO {4}'.format(self._as_who, self._schema, self._set_what, obj, self._for_whom))
            else:
                self.query.append('ALTER DEFAULT PRIVILEGES{0} GRANT {1} ON {2} TO {3}'.format(self._schema, self._set_what, obj, self._for_whom))
            self.add_grant_option()

    def build_present(self):
        if self._obj_type == 'default_privs':
            self.add_default_revoke()
            self.add_default_priv()
        else:
            self.query.append('GRANT {0} TO {1}'.format(self._set_what, self._for_whom))
            self.add_grant_option()

    def build_absent(self):
        if self._obj_type == 'default_privs':
            self.query = []
            for obj in ['TABLES', 'SEQUENCES', 'TYPES']:
                if self._as_who:
                    self.query.append('ALTER DEFAULT PRIVILEGES FOR ROLE {0}{1} REVOKE ALL ON {2} FROM {3};'.format(self._as_who, self._schema, obj, self._for_whom))
                else:
                    self.query.append('ALTER DEFAULT PRIVILEGES{0} REVOKE ALL ON {1} FROM {2};'.format(self._schema, obj, self._for_whom))
        else:
            self.query.append('REVOKE {0} FROM {1};'.format(self._set_what, self._for_whom))