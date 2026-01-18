import abc
import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
from requests import Response
from cinderclient.apiclient import exceptions
from cinderclient import utils
def _append_request_id(self, resp):
    if isinstance(resp, Response):
        request_id = resp.headers.get('x-openstack-request-id')
        self.x_openstack_request_ids.append(request_id)
    else:
        self.x_openstack_request_ids.append(resp)