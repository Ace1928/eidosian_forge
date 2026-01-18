import base64
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
class SecretFormatter(formatter.EntityFormatter):
    columns = ('Secret href', 'Name', 'Created', 'Status', 'Content types', 'Algorithm', 'Bit length', 'Secret type', 'Mode', 'Expiration')

    def _get_formatted_data(self):
        created = self.created.isoformat() if self.created else None
        expiration = self.expiration.isoformat() if self.expiration else None
        data = (self.secret_ref, self.name, created, self.status, self.content_types, self.algorithm, self.bit_length, self.secret_type, self.mode, expiration)
        return data