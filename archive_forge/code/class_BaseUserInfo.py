from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
class BaseUserInfo(provider_api.ProviderAPIMixin, object):

    @classmethod
    def create(cls, auth_payload, method_name):
        user_auth_info = cls()
        user_auth_info._validate_and_normalize_auth_data(auth_payload)
        user_auth_info.METHOD_NAME = method_name
        return user_auth_info

    def __init__(self):
        self.user_id = None
        self.user_ref = None
        self.METHOD_NAME = None

    def _assert_domain_is_enabled(self, domain_ref):
        try:
            PROVIDERS.resource_api.assert_domain_enabled(domain_id=domain_ref['id'], domain=domain_ref)
        except AssertionError as e:
            LOG.warning(e)
            raise exception.Unauthorized from e

    def _assert_user_is_enabled(self, user_ref):
        try:
            PROVIDERS.identity_api.assert_user_enabled(user_id=user_ref['id'], user=user_ref)
        except AssertionError as e:
            LOG.warning(e)
            raise exception.Unauthorized from e

    def _lookup_domain(self, domain_info):
        domain_id = domain_info.get('id')
        domain_name = domain_info.get('name')
        if not domain_id and (not domain_name):
            raise exception.ValidationError(attribute='id or name', target='domain')
        try:
            if domain_name:
                domain_ref = PROVIDERS.resource_api.get_domain_by_name(domain_name)
            else:
                domain_ref = PROVIDERS.resource_api.get_domain(domain_id)
        except exception.DomainNotFound as e:
            LOG.warning(e)
            raise exception.Unauthorized(e)
        self._assert_domain_is_enabled(domain_ref)
        return domain_ref

    def _validate_and_normalize_auth_data(self, auth_payload):
        if 'user' not in auth_payload:
            raise exception.ValidationError(attribute='user', target=self.METHOD_NAME)
        user_info = auth_payload['user']
        user_id = user_info.get('id')
        user_name = user_info.get('name')
        domain_ref = {}
        if not user_id and (not user_name):
            raise exception.ValidationError(attribute='id or name', target='user')
        try:
            if user_name:
                if 'domain' not in user_info:
                    raise exception.ValidationError(attribute='domain', target='user')
                domain_ref = self._lookup_domain(user_info['domain'])
                user_ref = PROVIDERS.identity_api.get_user_by_name(user_name, domain_ref['id'])
            else:
                user_ref = PROVIDERS.identity_api.get_user(user_id)
                domain_ref = PROVIDERS.resource_api.get_domain(user_ref['domain_id'])
                self._assert_domain_is_enabled(domain_ref)
        except exception.UserNotFound as e:
            LOG.warning(e)
            audit_reason = reason.Reason(str(e), str(e.code))
            audit_initiator = notifications.build_audit_initiator()
            if user_name:
                audit_initiator.user_name = user_name
            else:
                audit_initiator.user_id = user_id
            audit_initiator.domain_id = domain_ref.get('id')
            audit_initiator.domain_name = domain_ref.get('name')
            notifications._send_audit_notification(action=_NOTIFY_OP, initiator=audit_initiator, outcome=taxonomy.OUTCOME_FAILURE, target=resource.Resource(typeURI=taxonomy.ACCOUNT_USER), event_type=_NOTIFY_EVENT, reason=audit_reason)
            raise exception.Unauthorized(e)
        self._assert_user_is_enabled(user_ref)
        self.user_ref = user_ref
        self.user_id = user_ref['id']
        self.domain_id = domain_ref['id']