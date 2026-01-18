import abc
import argparse
import logging
import uuid
from oslo_config import cfg
from oslo_utils import timeutils
from stevedore import extension
from stevedore import named
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
def _sanitize_context(ctxt):
    if ctxt is None or type(ctxt) is dict:
        return {}
    try:
        return ctxt.redacted_copy()
    except AttributeError:
        _LOG.warning('Unable to properly redact context for notification, omitting context from notification.')
        return {}