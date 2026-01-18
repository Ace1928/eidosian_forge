import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _record_msg_type(record: 'pb.Record') -> str:
    msg_type = str(record.WhichOneof('record_type'))
    if msg_type == 'request':
        request = record.request
        msg_type = str(request.WhichOneof('request_type'))
    return msg_type