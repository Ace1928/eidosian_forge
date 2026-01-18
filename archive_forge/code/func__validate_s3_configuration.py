import copy
from botocore.compat import OrderedDict
from botocore.endpoint import DEFAULT_TIMEOUT, MAX_POOL_CONNECTIONS
from botocore.exceptions import (
def _validate_s3_configuration(self, s3):
    if s3 is not None:
        addressing_style = s3.get('addressing_style')
        if addressing_style not in ['virtual', 'auto', 'path', None]:
            raise InvalidS3AddressingStyleError(s3_addressing_style=addressing_style)