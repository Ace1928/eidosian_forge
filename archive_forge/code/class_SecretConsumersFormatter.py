import base64
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
class SecretConsumersFormatter(formatter.EntityFormatter):
    columns = ('Service', 'Resource type', 'Resource id', 'Created')

    def _get_formatted_data(self):
        data = (self.service, self.resource_type, self.resource_id, self.created)
        return data