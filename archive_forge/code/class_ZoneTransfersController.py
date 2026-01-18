from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class ZoneTransfersController(V2Controller):

    def create_request(self, zone, target_project_id, description=None):
        zone = v2_utils.resolve_by_name(self.client.zones.list, zone)
        data = {'target_project_id': target_project_id}
        if description is not None:
            data['description'] = description
        url = f'/zones/{zone}/tasks/transfer_requests'
        return self._post(url, data=data)

    def get_request(self, transfer_id):
        url = f'/zones/tasks/transfer_requests/{transfer_id}'
        return self._get(url)

    def list_requests(self):
        url = '/zones/tasks/transfer_requests'
        return self._get(url, response_key='transfer_requests')

    def update_request(self, transfer_id, values):
        url = f'/zones/tasks/transfer_requests/{transfer_id}'
        return self._patch(url, data=values)

    def delete_request(self, transfer_id):
        url = f'/zones/tasks/transfer_requests/{transfer_id}'
        self._delete(url)

    def accept_request(self, transfer_id, key):
        url = '/zones/tasks/transfer_accepts'
        data = {'key': key, 'zone_transfer_request_id': transfer_id}
        return self._post(url, data=data)

    def get_accept(self, accept_id):
        url = f'/zones/tasks/transfer_accepts/{accept_id}'
        return self._get(url)

    def list_accepts(self):
        url = '/zones/tasks/transfer_accepts'
        return self._get(url, response_key='transfer_accepts')