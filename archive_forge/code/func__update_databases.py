from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _update_databases(self, instance, databases):
    if databases:
        for db in databases:
            if db.get('ACTION') == self.CREATE:
                db.pop('ACTION', None)
                dmsg = 'Adding new database %(db)s to instance'
                LOG.debug(dmsg % {'db': db})
                self.client().databases.create(instance, [db])
            elif db.get('ACTION') == self.DELETE:
                dmsg = 'Deleting existing database %(db)s from instance'
                LOG.debug(dmsg % {'db': db['name']})
                self.client().databases.delete(instance, db['name'])
    return True