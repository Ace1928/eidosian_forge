import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def _validate_users_type(self):
    if self.users and (not (type(self.users) is list or type(self.users) is set)):
        raise ValueError('Users value is expected to be provided as list/set.')