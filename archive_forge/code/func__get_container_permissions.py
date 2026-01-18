import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
def _get_container_permissions(self, container_name):
    """
        Return the container permissions for the current authenticated user.

        :param container_name:  The container name.
        :param container_name: ``str``

        :return: The permissions on the container.
        :rtype: ``int`` from ContainerPermissions
        """
    try:
        self.json_connection.request('/storage/v1/b/%s/acl' % container_name)
        return ContainerPermissions.OWNER
    except ProviderError as e:
        if e.http_code == httplib.FORBIDDEN:
            pass
        elif e.http_code == httplib.NOT_FOUND:
            return ContainerPermissions.NONE
        else:
            raise
    try:
        self.json_connection.request('/storage/v1/b/%s/o/writecheck' % container_name, headers={'x-goog-if-generation-match': '0'}, method='DELETE')
    except ProviderError as e:
        if e.http_code in [httplib.NOT_FOUND, httplib.PRECONDITION_FAILED]:
            return ContainerPermissions.WRITER
        elif e.http_code != httplib.FORBIDDEN:
            raise
    try:
        self.json_connection.request('/storage/v1/b/%s' % container_name)
        return ContainerPermissions.READER
    except ProviderError as e:
        if e.http_code not in [httplib.FORBIDDEN, httplib.NOT_FOUND]:
            raise
    return ContainerPermissions.NONE