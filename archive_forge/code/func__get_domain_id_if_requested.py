from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
from osc_lib import utils
from neutronclient._i18n import _
def _get_domain_id_if_requested(identity_client, domain_name_or_id):
    if not domain_name_or_id:
        return None
    domain = find_domain(identity_client, domain_name_or_id)
    return domain.id