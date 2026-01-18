import logging
import ssl
import sys
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, Struct
class TransportBase:

    @classmethod
    def supports_feature(cls, feature_name):
        return cls._wrapper_name in _http_facilities[feature_name]