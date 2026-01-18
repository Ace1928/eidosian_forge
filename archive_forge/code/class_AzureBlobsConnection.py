import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class AzureBlobsConnection(AzureConnection):
    """
    Represents a single connection to Azure Blobs.

    The main Azure Blob Storage service uses a prefix in the hostname to
    distinguish between accounts, e.g. ``theaccount.blob.core.windows.net``.
    However, some custom deployments of the service, such as the Azurite
    emulator, instead use a URL prefix such as ``/theaccount``. To support
    these deployments, the parameter ``account_prefix`` must be set on the
    connection. This is done by instantiating the driver with arguments such
    as ``host='somewhere.tld'`` and ``key='theaccount'``. To specify a custom
    host without an account prefix, e.g. to connect to Azure Government or
    Azure China, the driver can be instantiated with the appropriate storage
    endpoint suffix, e.g. ``host='blob.core.usgovcloudapi.net'`` and
    ``key='theaccount'``.

    :param account_prefix: Optional prefix identifying the storage account.
                           Used when connecting to a custom deployment of the
                           storage service like Azurite or IoT Edge Storage.
    :type account_prefix: ``str``
    """

    def __init__(self, *args, **kwargs):
        self.account_prefix = kwargs.pop('account_prefix', None)
        super().__init__(*args, **kwargs)

    def morph_action_hook(self, action):
        action = super().morph_action_hook(action)
        if self.account_prefix is not None:
            action = '/{}{}'.format(self.account_prefix, action)
        return action
    API_VERSION = '2018-11-09'