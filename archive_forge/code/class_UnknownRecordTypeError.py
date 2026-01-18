import re
from io import BytesIO
from .. import errors
class UnknownRecordTypeError(ContainerError):
    _fmt = 'Unknown record type: %(record_type)r'

    def __init__(self, record_type):
        self.record_type = record_type