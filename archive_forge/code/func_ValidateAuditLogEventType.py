from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def ValidateAuditLogEventType(name):
    if not IsAuditLogType(name):
        raise InvalidEventType('For this command, the event type must be: {}.'.format(_AUDIT_LOG_TYPE.name))