import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def _get_operation_acl(self, operation_type):
    return next((acl for acl in self._operation_acls if acl.operation_type == operation_type), None)