import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def _fill_secrets_from_secret_refs(self):
    if self._secret_refs:
        self._cached_secrets = dict(((name.lower() if name else '', self._secret_manager.get(secret_ref=secret_ref)) for name, secret_ref in self._secret_refs.items()))