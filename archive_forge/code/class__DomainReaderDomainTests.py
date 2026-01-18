import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import domain as dp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _DomainReaderDomainTests(object):

    def test_user_can_list_domains(self):
        second_domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            r = c.get('/v3/domains', headers=self.headers)
            self.assertEqual(1, len(r.json['domains']))
            self.assertNotIn(second_domain['id'], [d['id'] for d in r.json['domains']])
            self.assertEqual(self.domain_id, r.json['domains'][0]['id'])

    def test_user_can_filter_domains_by_name(self):
        second_domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            r = c.get('/v3/domains?name=%s' % self.domain['name'], headers=self.headers)
            self.assertEqual(1, len(r.json['domains']))
            self.assertNotIn(second_domain['id'], [d['id'] for d in r.json['domains']])
            self.assertEqual(self.domain['id'], r.json['domains'][0]['id'])
            r = c.get('/v3/domains?name=%s' % second_domain['name'], headers=self.headers)
            self.assertEqual(0, len(r.json['domains']))

    def test_user_can_filter_domains_by_enabled(self):
        enabled_domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        disabled_domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref(enabled=False))
        with self.test_client() as c:
            r = c.get('/v3/domains?enabled=true', headers=self.headers)
            enabled_domain_ids = []
            for domain in r.json['domains']:
                enabled_domain_ids.append(domain['id'])
            self.assertEqual(1, len(r.json['domains']))
            self.assertEqual(self.domain_id, r.json['domains'][0]['id'])
            self.assertNotIn(enabled_domain['id'], enabled_domain_ids)
            self.assertNotIn(disabled_domain['id'], enabled_domain_ids)
            r = c.get('/v3/domains?enabled=false', headers=self.headers)
            self.assertEqual(0, len(r.json['domains']))