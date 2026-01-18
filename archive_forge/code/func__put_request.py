from openstack.exceptions import HttpException
from openstack import resource
from openstack import utils
def _put_request(self, session, url, json_data):
    resp = session.put(url, json=json_data)
    data = resp.json()
    if not resp.ok:
        message = None
        if 'NeutronError' in data:
            message = data['NeutronError']['message']
        raise HttpException(message=message, response=resp)
    self._body.attributes.update(data)
    self._update_location()
    return self