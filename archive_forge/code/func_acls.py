import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
@property
def acls(self):
    """Get ACL settings for this container."""
    if self._container_ref and (not self._acls):
        self._acls = self._acl_manager.get(self.container_ref)
    return self._acls