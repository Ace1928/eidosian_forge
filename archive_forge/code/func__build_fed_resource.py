import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _build_fed_resource(self):
    new_mapping = unit.new_mapping_ref()
    PROVIDERS.federation_api.create_mapping(new_mapping['id'], new_mapping)
    for idp_id, protocol_id in [('ORG_IDP', 'saml2'), ('myidp', 'mapped')]:
        new_idp = unit.new_identity_provider_ref(idp_id=idp_id, domain_id='default')
        new_protocol = unit.new_protocol_ref(protocol_id=protocol_id, idp_id=idp_id, mapping_id=new_mapping['id'])
        PROVIDERS.federation_api.create_idp(new_idp['id'], new_idp)
        PROVIDERS.federation_api.create_protocol(new_idp['id'], new_protocol['id'], new_protocol)