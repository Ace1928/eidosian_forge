import io
import json
import mimetypes
from sentry_sdk._compat import text_type, PY2
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions
def add_checkin(self, checkin):
    self.add_item(Item(payload=PayloadRef(json=checkin), type='check_in'))