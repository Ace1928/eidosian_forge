import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def _initialize_secrets(self, secrets):
    try:
        self._fill_secrets_from_secret_refs()
    except Exception:
        raise ValueError('One or more of the provided secret_refs could not be retrieved!')
    if secrets:
        try:
            for name, secret in secrets.items():
                self.add(name, secret)
        except Exception:
            raise ValueError('One or more of the provided secrets are not valid Secret objects!')