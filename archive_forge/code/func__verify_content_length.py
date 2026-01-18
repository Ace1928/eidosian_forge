import logging
from io import IOBase
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from botocore import parsers
from botocore.compat import set_socket_timeout
from botocore.exceptions import (
from botocore import ScalarTypes  # noqa
from botocore.compat import XMLParseError  # noqa
from botocore.hooks import first_non_none_response  # noqa
def _verify_content_length(self):
    if self._content_length is not None and self._amount_read != int(self._content_length):
        raise IncompleteReadError(actual_bytes=self._amount_read, expected_bytes=int(self._content_length))