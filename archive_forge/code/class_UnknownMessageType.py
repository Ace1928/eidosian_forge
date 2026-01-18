import functools
import inspect
import sys
import msgpack
import rapidjson
from ruamel import yaml
class UnknownMessageType(Exception):
    """Raised when trying to decode an unknown message type."""