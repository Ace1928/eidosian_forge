from __future__ import absolute_import
from __future__ import unicode_literals
import os
class QueueEntry(validation.Validated):
    """Describes a single task queue."""
    ATTRIBUTES = {NAME: _NAME_REGEX, RATE: validation.Optional(_RATE_REGEX), MODE: validation.Optional(_MODE_REGEX), BUCKET_SIZE: validation.Optional(validation.TYPE_INT), MAX_CONCURRENT_REQUESTS: validation.Optional(validation.TYPE_INT), RETRY_PARAMETERS: validation.Optional(RetryParameters), ACL: validation.Optional(validation.Repeated(Acl)), TARGET: validation.Optional(_VERSION_REGEX)}