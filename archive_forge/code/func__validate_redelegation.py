from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
@staticmethod
def _validate_redelegation(redelegated_trust, trust):
    max_redelegation_count = CONF.trust.max_redelegation_count
    redelegation_depth = redelegated_trust.get('redelegation_count', 0)
    if not 0 < redelegation_depth <= max_redelegation_count:
        raise exception.Forbidden(_('Remaining redelegation depth of %(redelegation_depth)d out of allowed range of [0..%(max_count)d]') % {'redelegation_depth': redelegation_depth, 'max_count': max_redelegation_count})
    remaining_uses = trust.get('remaining_uses')
    if remaining_uses is not None:
        raise exception.Forbidden(_('Field "remaining_uses" is set to %(value)s while it must not be set in order to redelegate a trust'), value=remaining_uses)
    trust_expiry = trust.get('expires_at')
    redelegated_expiry = redelegated_trust['expires_at']
    if trust_expiry:
        if redelegated_expiry < trust_expiry.replace(tzinfo=None):
            raise exception.Forbidden(_('Requested expiration time is more than redelegated trust can provide'))
    else:
        trust['expires_at'] = redelegated_expiry
    parent_roles = set((role['id'] for role in redelegated_trust['roles']))
    if not all((role['id'] in parent_roles for role in trust['roles'])):
        raise exception.Forbidden(_('Some of requested roles are not in redelegated trust'))
    if not redelegated_trust['impersonation'] and trust['impersonation']:
        raise exception.Forbidden(_('Impersonation is not allowed because redelegated trust does not specify impersonation. Redelegated trust id: %s') % redelegated_trust['id'])