import uuid
import ldap.filter
from oslo_log import log
from oslo_log import versionutils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends.ldap import models
class UserApi(common_ldap.EnabledEmuMixIn, common_ldap.BaseLdap):
    DEFAULT_OU = 'ou=Users'
    DEFAULT_STRUCTURAL_CLASSES = ['person']
    DEFAULT_ID_ATTR = 'cn'
    DEFAULT_OBJECTCLASS = 'inetOrgPerson'
    NotFound = exception.UserNotFound
    options_name = 'user'
    attribute_options_names = {'password': 'pass', 'email': 'mail', 'name': 'name', 'description': 'description', 'enabled': 'enabled', 'default_project_id': 'default_project_id'}
    immutable_attrs = ['id']
    model = models.User

    def __init__(self, conf):
        super(UserApi, self).__init__(conf)
        self.enabled_mask = conf.ldap.user_enabled_mask
        self.enabled_default = conf.ldap.user_enabled_default
        self.enabled_invert = conf.ldap.user_enabled_invert
        self.enabled_emulation = conf.ldap.user_enabled_emulation

    def _ldap_res_to_model(self, res):
        obj = super(UserApi, self)._ldap_res_to_model(res)
        if self.enabled_mask != 0:
            enabled = int(obj.get('enabled', self.enabled_default))
            obj['enabled'] = enabled & self.enabled_mask != self.enabled_mask
        elif self.enabled_invert and (not self.enabled_emulation):
            enabled = obj.get('enabled', self.enabled_default)
            if isinstance(enabled, str):
                if enabled.lower() == 'true':
                    enabled = True
                else:
                    enabled = False
            obj['enabled'] = not enabled
        obj['dn'] = res[0]
        return obj

    def mask_enabled_attribute(self, values):
        value = values['enabled']
        values.setdefault('enabled_nomask', int(self.enabled_default))
        if value != (values['enabled_nomask'] & self.enabled_mask != self.enabled_mask):
            values['enabled_nomask'] ^= self.enabled_mask
        values['enabled'] = values['enabled_nomask']
        del values['enabled_nomask']

    def create(self, values):
        if 'options' in values:
            values.pop('options')
        if self.enabled_mask:
            orig_enabled = values['enabled']
            self.mask_enabled_attribute(values)
        elif self.enabled_invert and (not self.enabled_emulation):
            orig_enabled = values['enabled']
            if orig_enabled is not None:
                values['enabled'] = not orig_enabled
            else:
                values['enabled'] = self.enabled_default
        values = super(UserApi, self).create(values)
        if self.enabled_mask or (self.enabled_invert and (not self.enabled_emulation)):
            values['enabled'] = orig_enabled
        values['options'] = {}
        return values

    def get(self, user_id, ldap_filter=None):
        obj = super(UserApi, self).get(user_id, ldap_filter=ldap_filter)
        obj['options'] = {}
        return obj

    def get_filtered(self, user_id):
        try:
            user = self.get(user_id)
            return self.filter_attributes(user)
        except ldap.NO_SUCH_OBJECT:
            raise self.NotFound(user_id=user_id)

    def get_all(self, ldap_filter=None, hints=None):
        objs = super(UserApi, self).get_all(ldap_filter=ldap_filter, hints=hints)
        for obj in objs:
            obj['options'] = {}
        return objs

    def get_all_filtered(self, hints):
        query = self.filter_query(hints, self.ldap_filter)
        return [self.filter_attributes(user) for user in self.get_all(query, hints)]

    def filter_attributes(self, user):
        return base.filter_user(common_ldap.filter_entity(user))

    def is_user(self, dn):
        """Return True if the entry is a user."""
        return common_ldap.dn_startswith(dn, self.tree_dn)

    def update(self, user_id, values, old_obj=None):
        if old_obj is None:
            old_obj = self.get(user_id)
        if 'options' in old_obj:
            old_obj.pop('options')
        if 'options' in values:
            values.pop('options')
        values = super(UserApi, self).update(user_id, values, old_obj)
        values['options'] = {}
        return values