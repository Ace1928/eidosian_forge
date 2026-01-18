import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def _immutable_after_save(func):

    @functools.wraps(func)
    def wrapper(self, *args):
        if hasattr(self, '_container_ref') and self._container_ref:
            raise base.ImmutableException()
        return func(self, *args)
    return wrapper