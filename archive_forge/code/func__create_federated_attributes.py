import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def _create_federated_attributes(self):
    idp = {'id': uuid.uuid4().hex, 'enabled': True, 'description': uuid.uuid4().hex}
    PROVIDERS.federation_api.create_idp(idp['id'], idp)
    mapping = mapping_fixtures.MAPPING_EPHEMERAL_USER
    mapping['id'] = uuid.uuid4().hex
    PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)
    protocol = {'id': uuid.uuid4().hex, 'mapping_id': mapping['id']}
    PROVIDERS.federation_api.create_protocol(idp['id'], protocol['id'], protocol)
    return (idp, protocol)