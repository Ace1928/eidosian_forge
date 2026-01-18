from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import stack_user
class AccessKey(resource.Resource):
    PROPERTIES = SERIAL, USER_NAME, STATUS = ('Serial', 'UserName', 'Status')
    ATTRIBUTES = USER_NAME, SECRET_ACCESS_KEY = ('UserName', 'SecretAccessKey')
    properties_schema = {SERIAL: properties.Schema(properties.Schema.INTEGER, _('Not Implemented.'), implemented=False), USER_NAME: properties.Schema(properties.Schema.STRING, _('The name of the user that the new key will belong to.'), required=True), STATUS: properties.Schema(properties.Schema.STRING, _('Not Implemented.'), constraints=[constraints.AllowedValues(['Active', 'Inactive'])], implemented=False)}
    attributes_schema = {USER_NAME: attributes.Schema(_('Username associated with the AccessKey.'), cache_mode=attributes.Schema.CACHE_NONE, type=attributes.Schema.STRING), SECRET_ACCESS_KEY: attributes.Schema(_('Keypair secret key.'), cache_mode=attributes.Schema.CACHE_NONE, type=attributes.Schema.STRING)}

    def __init__(self, name, json_snippet, stack):
        super(AccessKey, self).__init__(name, json_snippet, stack)
        self._secret = None
        if self.resource_id:
            self._register_access_key()

    def _get_user(self):
        """Derive the keystone userid, stored in the User resource_id.

        Helper function to derive the keystone userid, which is stored in the
        resource_id of the User associated with this key. We want to avoid
        looking the name up via listing keystone users, as this requires admin
        rights in keystone, so FnGetAtt which calls _secret_accesskey won't
        work for normal non-admin users.
        """
        return self.stack.resource_by_refid(self.properties[self.USER_NAME])

    def handle_create(self):
        user = self._get_user()
        if user is None:
            raise exception.NotFound(_('could not find user %s') % self.properties[self.USER_NAME])
        kp = user._create_keypair()
        self.resource_id_set(kp.access)
        self._secret = kp.secret
        self._register_access_key()
        self.data_set('secret_key', kp.secret, redact=True)
        self.data_set('credential_id', kp.id, redact=True)

    def handle_delete(self):
        self._secret = None
        if self.resource_id is None:
            return
        user = self._get_user()
        if user is None:
            LOG.debug('Error deleting %s - user not found', str(self))
            return
        user._delete_keypair()

    def _secret_accesskey(self):
        """Return the user's access key.

        Fetching it from keystone if necessary.
        """
        if self._secret is None:
            if not self.resource_id:
                LOG.info('could not get secret for %(username)s Error:%(msg)s', {'username': self.properties[self.USER_NAME], 'msg': 'resource_id not yet set'})
            else:
                self._secret = self.data().get('secret_key')
                if self._secret is None:
                    try:
                        user_id = self._get_user().resource_id
                        kp = self.keystone().get_ec2_keypair(user_id=user_id, access=self.resource_id)
                        self._secret = kp.secret
                        self.data_set('secret_key', kp.secret, redact=True)
                        self.data_set('credential_id', kp.id, redact=True)
                    except Exception as ex:
                        LOG.info('could not get secret for %(username)s Error:%(msg)s', {'username': self.properties[self.USER_NAME], 'msg': ex})
        return self._secret or '000-000-000'

    def _resolve_attribute(self, name):
        if name == self.USER_NAME:
            return self.properties[self.USER_NAME]
        elif name == self.SECRET_ACCESS_KEY:
            return self._secret_accesskey()

    def _register_access_key(self):

        def access_allowed(resource_name):
            return self._get_user().access_allowed(resource_name)
        self.stack.register_access_allowed_handler(self.resource_id, access_allowed)