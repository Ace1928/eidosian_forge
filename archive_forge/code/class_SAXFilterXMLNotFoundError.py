import os
import sys
import re
from xml.sax.handler import ContentHandler
from ncclient.transport.errors import NetconfFramingError
from ncclient.transport.session import NetconfBase
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.operations.errors import OperationError
from ncclient.transport import SessionListener
import logging
class SAXFilterXMLNotFoundError(OperationError):

    def __init__(self, rpc_listener):
        self._listener = rpc_listener

    def __str__(self):
        return 'SAX filter input xml not provided for listener: %s' % self._listener