from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _validate_token_user(self):
    if self.trust_scoped:
        if self.user_id != self.trustee['id']:
            raise exception.Forbidden(_('User is not a trustee.'))
        try:
            PROVIDERS.resource_api.assert_domain_enabled(self.trustor['domain_id'])
        except AssertionError:
            raise exception.TokenNotFound(_('Trustor domain is disabled.'))
        try:
            PROVIDERS.resource_api.assert_domain_enabled(self.trustee['domain_id'])
        except AssertionError:
            raise exception.TokenNotFound(_('Trustee domain is disabled.'))
        try:
            PROVIDERS.identity_api.assert_user_enabled(self.trustor['id'])
        except AssertionError:
            raise exception.Forbidden(_('Trustor is disabled.'))
    if not self.user_domain.get('enabled'):
        msg = 'Unable to validate token because domain %(id)s is disabled' % {'id': self.user_domain['id']}
        tr_msg = _('Unable to validate token because domain %(id)s is disabled') % {'id': self.user_domain['id']}
        LOG.warning(msg)
        raise exception.DomainNotFound(tr_msg)