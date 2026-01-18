import abc
import base64
import enum
import json
import six
from google.auth import exceptions
class ClientAuthType(enum.Enum):
    basic = 1
    request_body = 2