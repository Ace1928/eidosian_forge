from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgTablespace(object):
    """Class for working with PostgreSQL tablespaces.

    Args:
        module (AnsibleModule) -- object of AnsibleModule class
        cursor (cursor) -- cursor object of psycopg library
        name (str) -- name of the tablespace

    Attrs:
        module (AnsibleModule) -- object of AnsibleModule class
        cursor (cursor) -- cursor object of psycopg library
        name (str) -- name of the tablespace
        exists (bool) -- flag the tablespace exists in the DB or not
        owner (str) -- tablespace owner
        location (str) -- path to the tablespace directory in the file system
        executed_queries (list) -- list of executed queries
        new_name (str) -- new name for the tablespace
        opt_not_supported (bool) -- flag indicates a tablespace option is supported or not
    """

    def __init__(self, module, cursor, name):
        self.module = module
        self.cursor = cursor
        self.name = name
        self.exists = False
        self.owner = ''
        self.settings = {}
        self.location = ''
        self.executed_queries = []
        self.new_name = ''
        self.opt_not_supported = False
        self.comment = None
        self.get_info()

    def get_info(self):
        """Get tablespace information."""
        opt = exec_sql(self, "SELECT 1 FROM information_schema.columns WHERE table_name = 'pg_tablespace' AND column_name = 'spcoptions'", add_to_executed=False)
        location = exec_sql(self, "SELECT 1 FROM information_schema.columns WHERE table_name = 'pg_tablespace' AND column_name = 'spclocation'", add_to_executed=False)
        if location:
            location = 'spclocation'
        else:
            location = 'pg_tablespace_location(t.oid)'
        if not opt:
            self.opt_not_supported = True
            query = "SELECT shobj_description(t.oid, 'pg_tablespace') AS comment, r.rolname, (SELECT Null) spcoptions, %s loc_string FROM pg_catalog.pg_tablespace AS t JOIN pg_catalog.pg_roles AS r ON t.spcowner = r.oid " % location
        else:
            query = "SELECT shobj_description(t.oid, 'pg_tablespace') AS comment, r.rolname, t.spcoptions, %s loc_string FROM pg_catalog.pg_tablespace AS t JOIN pg_catalog.pg_roles AS r ON t.spcowner = r.oid " % location
        res = exec_sql(self, query + 'WHERE t.spcname = %(name)s', query_params={'name': self.name}, add_to_executed=False)
        if not res:
            self.exists = False
            return False
        if res[0]['rolname']:
            self.exists = True
            self.owner = res[0]['rolname']
            if res[0]['spcoptions']:
                for i in res[0]['spcoptions']:
                    i = i.split('=')
                    self.settings[i[0]] = i[1]
            if res[0]['loc_string']:
                self.location = res[0]['loc_string']
            self.comment = res[0]['comment'] if res[0]['comment'] is not None else ''

    def create(self, location):
        """Create tablespace.

        Return True if success, otherwise, return False.

        args:
            location (str) -- tablespace directory path in the FS
        """
        query = 'CREATE TABLESPACE "%s" LOCATION \'%s\'' % (self.name, location)
        return exec_sql(self, query, return_bool=True)

    def drop(self):
        """Drop tablespace.

        Return True if success, otherwise, return False.
        """
        return exec_sql(self, 'DROP TABLESPACE "%s"' % self.name, return_bool=True)

    def set_owner(self, new_owner):
        """Set tablespace owner.

        Return True if success, otherwise, return False.

        args:
            new_owner (str) -- name of a new owner for the tablespace"
        """
        if new_owner == self.owner:
            return False
        query = 'ALTER TABLESPACE "%s" OWNER TO "%s"' % (self.name, new_owner)
        return exec_sql(self, query, return_bool=True)

    def set_comment(self, comment, check_mode):
        """Set tablespace comment.

        Return True if success, otherwise, return False.

        args:
            comment (str) -- comment to set for the tablespace"
        """
        if comment == self.comment:
            return False
        return set_comment(self.cursor, comment, 'tablespace', self.name, check_mode, self.executed_queries)

    def rename(self, newname):
        """Rename tablespace.

        Return True if success, otherwise, return False.

        args:
            newname (str) -- new name for the tablespace"
        """
        query = 'ALTER TABLESPACE "%s" RENAME TO "%s"' % (self.name, newname)
        self.new_name = newname
        return exec_sql(self, query, return_bool=True)

    def set_settings(self, new_settings):
        """Set tablespace settings (options).

        If some setting has been changed, set changed = True.
        After all settings list is handling, return changed.

        args:
            new_settings (list) -- list of new settings
        """
        if self.opt_not_supported:
            return False
        changed = False
        for i in new_settings:
            if new_settings[i] == 'reset':
                if i in self.settings:
                    changed = self.__reset_setting(i)
                    self.settings[i] = None
            elif i not in self.settings or str(new_settings[i]) != self.settings[i]:
                changed = self.__set_setting("%s = '%s'" % (i, new_settings[i]))
        return changed

    def __reset_setting(self, setting):
        """Reset tablespace setting.

        Return True if success, otherwise, return False.

        args:
            setting (str) -- string in format "setting_name = 'setting_value'"
        """
        query = 'ALTER TABLESPACE "%s" RESET (%s)' % (self.name, setting)
        return exec_sql(self, query, return_bool=True)

    def __set_setting(self, setting):
        """Set tablespace setting.

        Return True if success, otherwise, return False.

        args:
            setting (str) -- string in format "setting_name = 'setting_value'"
        """
        query = 'ALTER TABLESPACE "%s" SET (%s)' % (self.name, setting)
        return exec_sql(self, query, return_bool=True)