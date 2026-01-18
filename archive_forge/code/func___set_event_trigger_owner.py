from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __set_event_trigger_owner(self):
    """Set the event trigger owner."""
    query = 'ALTER EVENT TRIGGER %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
    self.changed = exec_sql(self, query, return_bool=True)