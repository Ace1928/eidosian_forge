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
def _build_keyring_key(self, **kwargs):
    """Create a unique key for keyring.

        Used to store and retrieve auth_ref from keyring.

        Return a slash-separated string of values ordered by key name.

        """
    return '/'.join([kwargs[k] or '?' for k in sorted(kwargs)])