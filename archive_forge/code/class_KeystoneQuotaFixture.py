import logging as std_logging
import os
from unittest import mock
import warnings
import fixtures as pyfixtures
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from oslo_db import warning as oslo_db_warning
from oslo_limit import limit
from sqlalchemy import exc as sqla_exc
class KeystoneQuotaFixture(pyfixtures.Fixture):

    def __init__(self, **defaults):
        self.defaults = defaults

    def setUp(self):
        super(KeystoneQuotaFixture, self).setUp()
        self.mock_conn = mock.MagicMock()
        limit._SDK_CONNECTION = self.mock_conn
        mock_gem = self.useFixture(pyfixtures.MockPatch('oslo_limit.limit.Enforcer._get_enforcement_model')).mock
        mock_gem.return_value = 'flat'
        fake_endpoint = endpoint.Endpoint()
        fake_endpoint.service_id = 'service_id'
        fake_endpoint.region_id = 'region_id'
        self.mock_conn.get_endpoint.return_value = fake_endpoint

        def fake_limits(service_id, region_id, resource_name, project_id):
            this_limit = klimit.Limit()
            this_limit.resource_name = resource_name
            this_limit.resource_limit = self.defaults[resource_name]
            return iter([this_limit])
        self.mock_conn.limits.side_effect = fake_limits