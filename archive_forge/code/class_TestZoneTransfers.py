import time
import uuid
from designateclient.tests import v2
class TestZoneTransfers(v2.APIV2TestCase, v2.CrudMixin):

    def test_create_request(self):
        zone = '098bee04-fe30-4a83-8ccd-e0c496755816'
        project = '123'
        ref = {'target_project_id': project}
        parts = ['zones', zone, 'tasks', 'transfer_requests']
        self.stub_url('POST', parts=parts, json=ref)
        self.client.zone_transfers.create_request(zone, project)
        self.assertRequestBodyIs(json=ref)

    def test_create_request_with_description(self):
        zone = '098bee04-fe30-4a83-8ccd-e0c496755816'
        project = '123'
        ref = {'target_project_id': project, 'description': 'My Foo'}
        parts = ['zones', zone, 'tasks', 'transfer_requests']
        self.stub_url('POST', parts=parts, json=ref)
        self.client.zone_transfers.create_request(zone, project, ref['description'])
        self.assertRequestBodyIs(json=ref)

    def test_get_request(self):
        transfer = '098bee04-fe30-4a83-8ccd-e0c496755816'
        project = '098bee04-fe30-4a83-8ccd-e0c496755817'
        ref = {'target_project_id': project}
        parts = ['zones', 'tasks', 'transfer_requests', transfer]
        self.stub_url('GET', parts=parts, json=ref)
        self.client.zone_transfers.get_request(transfer)
        self.assertRequestBodyIs('')

    def test_list_request(self):
        project = '098bee04-fe30-4a83-8ccd-e0c496755817'
        ref = [{'target_project_id': project}]
        parts = ['zones', 'tasks', 'transfer_requests']
        self.stub_url('GET', parts=parts, json={'transfer_requests': ref})
        self.client.zone_transfers.list_requests()
        self.assertRequestBodyIs('')

    def test_update_request(self):
        transfer = '098bee04-fe30-4a83-8ccd-e0c496755816'
        project = '098bee04-fe30-4a83-8ccd-e0c496755817'
        ref = {'target_project_id': project}
        parts = ['zones', 'tasks', 'transfer_requests', transfer]
        self.stub_url('PATCH', parts=parts, json=ref)
        self.client.zone_transfers.update_request(transfer, ref)
        self.assertRequestBodyIs(json=ref)

    def test_delete_request(self):
        transfer = '098bee04-fe30-4a83-8ccd-e0c496755816'
        parts = ['zones', 'tasks', 'transfer_requests', transfer]
        self.stub_url('DELETE', parts=parts)
        self.client.zone_transfers.delete_request(transfer)
        self.assertRequestBodyIs('')

    def test_accept_request(self):
        transfer = '098bee04-fe30-4a83-8ccd-e0c496755816'
        key = 'foo123'
        ref = {'status': 'COMPLETE'}
        parts = ['zones', 'tasks', 'transfer_accepts']
        self.stub_url('POST', parts=parts, json=ref)
        request = {'key': key, 'zone_transfer_request_id': transfer}
        self.client.zone_transfers.accept_request(transfer, key)
        self.assertRequestBodyIs(json=request)

    def test_get_accept(self):
        accept_id = '098bee04-fe30-4a83-8ccd-e0c496755816'
        ref = {'status': 'COMPLETE'}
        parts = ['zones', 'tasks', 'transfer_accepts', accept_id]
        self.stub_url('GET', parts=parts, json=ref)
        response = self.client.zone_transfers.get_accept(accept_id)
        self.assertEqual(ref, response)

    def test_list_accepts(self):
        accept_id = '098bee04-fe30-4a83-8ccd-e0c496755816'
        ref = {'id': accept_id, 'status': 'COMPLETE'}
        parts = ['zones', 'tasks', 'transfer_accepts']
        self.stub_url('GET', parts=parts, json={'transfer_accepts': ref})
        self.client.zone_transfers.list_accepts()
        self.assertRequestBodyIs('')