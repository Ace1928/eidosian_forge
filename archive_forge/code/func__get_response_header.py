from pecan.middleware.static import (StaticFileMiddleware, FileWrapper,
from pecan.tests import PecanTestCase
import os
def _get_response_header(self, header):
    for k, v in self._response_headers:
        if k.upper() == header.upper():
            return v
    return None