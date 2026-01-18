from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import requests
from blazarclient import exception
from blazarclient.i18n import _
class RequestManager(object):
    """Manager to create request from given Blazar URL and auth token."""

    def __init__(self, blazar_url, auth_token, user_agent):
        self.blazar_url = blazar_url
        self.auth_token = auth_token
        self.user_agent = user_agent

    def get(self, url):
        """Sends get request to Blazar.

        :param url: URL to the wanted Blazar resource.
        :type url: str
        """
        return self.request(url, 'GET')

    def post(self, url, body):
        """Sends post request to Blazar.

        :param url: URL to the wanted Blazar resource.
        :type url: str

        :param body: Values resource to be created from.
        :type body: dict
        """
        return self.request(url, 'POST', body=body)

    def delete(self, url):
        """Sends delete request to Blazar.

        :param url: URL to the wanted Blazar resource.
        :type url: str
        """
        return self.request(url, 'DELETE')

    def put(self, url, body):
        """Sends update request to Blazar.

        :param url: URL to the wanted Blazar resource.
        :type url: str

        :param body: Values resource to be updated from.
        :type body: dict
        """
        return self.request(url, 'PUT', body=body)

    def patch(self, url, body):
        """Sends patch request to Blazar.

        :param url: URL to the wanted Blazar resource.
        :type url: str
        """
        return self.request(url, 'PATCH', body=body)

    def request(self, url, method, **kwargs):
        """Base request method.

        Adds specific headers and URL prefix to the request.

        :param url: Resource URL.
        :type url: str

        :param method: Method to be called (GET, POST, PUT, DELETE).
        :type method: str

        :returns: Response and body.
        :rtype: tuple
        """
        kwargs.setdefault('headers', kwargs.get('headers', {}))
        kwargs['headers']['User-Agent'] = self.user_agent
        kwargs['headers']['Accept'] = 'application/json'
        kwargs['headers']['x-auth-token'] = self.auth_token
        if 'body' in kwargs:
            kwargs['headers']['Content-Type'] = 'application/json'
            kwargs['data'] = jsonutils.dump_as_bytes(kwargs['body'])
            del kwargs['body']
        resp = requests.request(method, self.blazar_url + url, **kwargs)
        try:
            body = jsonutils.loads(resp.text)
        except ValueError:
            body = None
        if resp.status_code >= 400:
            if body is not None:
                error_message = body.get('error_message', body)
            else:
                error_message = resp.text
            body = _('ERROR: {0}').format(error_message)
            raise exception.BlazarClientException(body, code=resp.status_code)
        return (resp, body)