import sys
import math
from datetime import datetime
from sentry_sdk.utils import (
from sentry_sdk._compat import (
from sentry_sdk._types import TYPE_CHECKING
def _is_request_body():
    try:
        if path[0] == 'request' and path[1] == 'data':
            return True
    except IndexError:
        return None
    return False