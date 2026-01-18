from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __transform_options(self):
    """Transform options dict into a suitable string."""
    for key, val in iteritems(self.module.params['options']):
        if key.upper() in self.opt_need_quotes:
            self.module.params['options'][key] = "'%s'" % val
    opt = ['%s %s' % (key, val) for key, val in iteritems(self.module.params['options'])]
    return '(%s)' % ', '.join(opt)