from __future__ import absolute_import
import json
from copy import deepcopy
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import AnnotatedValue
from sentry_sdk._compat import text_type, iteritems
from sentry_sdk._types import TYPE_CHECKING
def _is_json_content_type(ct):
    mt = (ct or '').split(';', 1)[0]
    return mt == 'application/json' or (mt.startswith('application/') and mt.endswith('+json'))