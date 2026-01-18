import importlib.metadata
import logging
import warnings
from debtcollector import removals
from debtcollector import renames
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import packaging.version
import requests
from keystoneclient import _discover
from keystoneclient import access
from keystoneclient.auth import base
from keystoneclient import baseclient
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
class _FakeRequestSession(object):
    """This object is a temporary hack that should be removed later.

    Keystoneclient has a cyclical dependency with its managers which is
    preventing it from being cleaned up correctly. This is always bad but when
    we switched to doing connection pooling this object wasn't getting cleaned
    either and so we had left over TCP connections hanging around.

    Until we can fix the client cleanup we rollback the use of a requests
    session and do individual connections like we used to.
    """

    def request(self, *args, **kwargs):
        return requests.request(*args, **kwargs)