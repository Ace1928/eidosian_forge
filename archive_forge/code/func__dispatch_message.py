import logging
import socket
import sys
import threading
from socket import AF_INET, SOCK_STREAM
from ssl import CERT_REQUIRED, SSLContext, SSLError
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TLSError
from ncclient.transport.session import Session
from ncclient.transport.parser import DefaultXMLParser
def _dispatch_message(self, raw):
    self.logger.info('Received message from host')
    self.logger.debug('Received:\n%s', raw)
    return super(TLSSession, self)._dispatch_message(raw)