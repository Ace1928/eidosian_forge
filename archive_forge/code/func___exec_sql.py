from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __exec_sql(self, query, ddl=False):
    """Execute SQL.

        Arguments:
            ddl (bool): If True, return True or False.
                Used for queries that don't return any rows
                (mainly for DDL queries) (default False).
        """
    try:
        self.cursor.execute(query)
        if not ddl:
            res = self.cursor.fetchall()
            return res
        return True
    except Exception as e:
        self.module.fail_json(msg="Cannot execute SQL '%s': %s" % (query, to_native(e)))
    return False