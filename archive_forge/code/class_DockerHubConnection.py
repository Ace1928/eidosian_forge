from base64 import b64encode
from libcloud.common.base import Connection, JsonResponse
from libcloud.container.base import ContainerImage
class DockerHubConnection(Connection):
    responseCls = JsonResponse

    def __init__(self, host, username=None, password=None, secure=True, port=None, url=None, timeout=None, proxy_url=None, backoff=None, retry_delay=None):
        super().__init__(secure=secure, host=host, port=port, url=url, timeout=timeout, proxy_url=proxy_url, backoff=backoff, retry_delay=retry_delay)
        self.username = username
        self.password = password

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'application/json'
        if self.username is not None:
            authstr = 'Basic ' + str(b64encode('{}:{}'.format(self.username, self.password).encode('latin1')).strip())
            headers['Authorization'] = authstr
        return headers