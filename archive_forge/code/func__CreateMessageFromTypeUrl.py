import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def _CreateMessageFromTypeUrl(type_url, descriptor_pool):
    """Creates a message from a type URL."""
    db = symbol_database.Default()
    pool = db.pool if descriptor_pool is None else descriptor_pool
    type_name = type_url.split('/')[-1]
    try:
        message_descriptor = pool.FindMessageTypeByName(type_name)
    except KeyError as e:
        raise TypeError('Can not find message descriptor by type_url: {0}'.format(type_url)) from e
    message_class = message_factory.GetMessageClass(message_descriptor)
    return message_class()