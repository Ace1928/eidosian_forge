import abc
import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
from requests import Response
from cinderclient.apiclient import exceptions
from cinderclient import utils
class RequestIdMixin(object):
    """Wrapper class to expose x-openstack-request-id to the caller."""

    def setup(self):
        self.x_openstack_request_ids = []

    @property
    def request_ids(self):
        return self.x_openstack_request_ids

    def append_request_ids(self, resp):
        """Add request_ids as an attribute to the object

        :param resp: list, Response object or string
        """
        if resp is None:
            return
        if isinstance(resp, list):
            for resp_obj in resp:
                self._append_request_id(resp_obj)
        else:
            self._append_request_id(resp)

    def _append_request_id(self, resp):
        if isinstance(resp, Response):
            request_id = resp.headers.get('x-openstack-request-id')
            self.x_openstack_request_ids.append(request_id)
        else:
            self.x_openstack_request_ids.append(resp)