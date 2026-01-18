from troveclient import base
from troveclient import common
from troveclient.v1 import users
def _enable_root(self, uri, root_password=None):
    """Implements root-enable API.
        Enable the root user and return the root password for the
        specified db instance or cluster.
        """
    if root_password:
        resp, body = self.api.client.post(uri, body={'password': root_password})
    else:
        resp, body = self.api.client.post(uri)
    common.check_for_exceptions(resp, body, uri)
    return (body['user']['name'], body['user']['password'])