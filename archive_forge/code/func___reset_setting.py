from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __reset_setting(self, setting):
    """Reset tablespace setting.

        Return True if success, otherwise, return False.

        args:
            setting (str) -- string in format "setting_name = 'setting_value'"
        """
    query = 'ALTER TABLESPACE "%s" RESET (%s)' % (self.name, setting)
    return exec_sql(self, query, return_bool=True)