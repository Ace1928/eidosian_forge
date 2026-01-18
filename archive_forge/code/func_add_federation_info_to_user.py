import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def add_federation_info_to_user(self, identity_provider=None, protocol=None, groups=None):
    data = {'OS-FEDERATION': {'identity_provider': identity_provider or uuid.uuid4().hex, 'protocol': protocol or uuid.uuid4().hex, 'groups': groups or [{'id': uuid.uuid4().hex}]}}
    self._user.update(data)
    return data