from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def accept_request(self, transfer_id, key):
    url = '/zones/tasks/transfer_accepts'
    data = {'key': key, 'zone_transfer_request_id': transfer_id}
    return self._post(url, data=data)