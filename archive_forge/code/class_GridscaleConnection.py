from libcloud.utils.py3 import httplib
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
class GridscaleConnection(ConnectionUserAndKey, PollingConnection):
    """
    gridscale connection class
    Authentication using uuid and api token

    """
    host = 'api.gridscale.io'
    responseCls = GridscaleResponse

    def encode_data(self, data):
        return json.dumps(data)

    def add_default_headers(self, headers):
        """
        add parameters that are necessary for each request to be successful

        :param headers: Authentication token
        :type headers: ``str``
        :return: None
        """
        headers['X-Auth-UserId'] = self.user_id
        headers['X-Auth-Token'] = self.key
        headers['Content-Type'] = 'application/json'
        return headers

    def async_request(self, *poargs, **kwargs):
        self.async_request_counter = 0
        self.request_method = '_poll_request_initial'
        return super().async_request(*poargs, **kwargs)

    def _poll_request_initial(self, **kwargs):
        if self.async_request_counter == 0:
            self.poll_response_initial = super().request(**kwargs)
            r = self.poll_response_initial
            self.async_request_counter += 1
        else:
            r = self.request(**kwargs)
        return r

    def get_poll_request_kwargs(self, response, context, request_kwargs):
        endpoint_url = 'requests/{}'.format(response.object['request_uuid'])
        kwargs = {'action': endpoint_url}
        return kwargs

    def has_completed(self, response):
        if response.status == 200:
            request_uuid = self.poll_response_initial.object['request_uuid']
            request_status = response.object[request_uuid]['status']
            if request_status == 'done':
                return True
        return False