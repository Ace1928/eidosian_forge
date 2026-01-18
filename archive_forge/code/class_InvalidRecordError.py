import re
from io import BytesIO
from .. import errors
class InvalidRecordError(ContainerError):
    _fmt = 'Invalid record: %(reason)s'

    def __init__(self, reason):
        self.reason = reason