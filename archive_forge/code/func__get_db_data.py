from __future__ import absolute_import
import copy
from sentry_sdk import Hub
from sentry_sdk.consts import SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import Span
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def _get_db_data(event):
    data = {}
    data[SPANDATA.DB_SYSTEM] = 'mongodb'
    db_name = event.database_name
    if db_name is not None:
        data[SPANDATA.DB_NAME] = db_name
    server_address = event.connection_id[0]
    if server_address is not None:
        data[SPANDATA.SERVER_ADDRESS] = server_address
    server_port = event.connection_id[1]
    if server_port is not None:
        data[SPANDATA.SERVER_PORT] = server_port
    return data