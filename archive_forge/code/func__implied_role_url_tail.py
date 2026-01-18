from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _implied_role_url_tail(self, prior_role, implied_role):
    base_url = '/%(prior_role_id)s/implies/%(implied_role_id)s' % {'prior_role_id': base.getid(prior_role), 'implied_role_id': base.getid(implied_role)}
    return base_url