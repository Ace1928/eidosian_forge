from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _update_users(self, instance, users):
    if users:
        for usr in users:
            dbs = [{'name': db} for db in usr.get(self.USER_DATABASES, [])]
            usr[self.USER_DATABASES] = dbs
            if usr.get('ACTION') == self.CREATE:
                usr.pop('ACTION', None)
                dmsg = 'Adding new user %(u)s to instance'
                LOG.debug(dmsg % {'u': usr})
                self.client().users.create(instance, [usr])
            elif usr.get('ACTION') == self.DELETE:
                dmsg = 'Deleting existing user %(u)s from instance'
                LOG.debug(dmsg % {'u': usr['name']})
                self.client().users.delete(instance, usr['name'])
            else:
                newattrs = {}
                if usr.get(self.USER_HOST):
                    newattrs[self.USER_HOST] = usr[self.USER_HOST]
                if usr.get(self.USER_PASSWORD):
                    newattrs[self.USER_PASSWORD] = usr[self.USER_PASSWORD]
                if newattrs:
                    self.client().users.update_attributes(instance, usr['name'], newuserattr=newattrs, hostname=instance.hostname)
                current = self.client().users.get(instance, usr[self.USER_NAME])
                dbs = [db['name'] for db in current.databases]
                desired = [db['name'] for db in usr.get(self.USER_DATABASES, [])]
                grants = [db for db in desired if db not in dbs]
                revokes = [db for db in dbs if db not in desired]
                if grants:
                    self.client().users.grant(instance, usr[self.USER_NAME], grants)
                if revokes:
                    self.client().users.revoke(instance, usr[self.USER_NAME], revokes)
    return True