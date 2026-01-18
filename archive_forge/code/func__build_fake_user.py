import testtools
from unittest import mock
from troveclient.v1 import users
def _build_fake_user(self, name, hostname=None, password=None, databases=None):
    return {'name': name, 'password': password if password else 'password', 'host': hostname, 'databases': databases if databases else []}