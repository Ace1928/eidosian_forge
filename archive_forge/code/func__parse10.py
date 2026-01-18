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
def _parse10(self):
    """Messages are delimited by MSG_DELIM. The buffer could have grown by
        a maximum of BUF_SIZE bytes everytime this method is called. Retains
        state across method calls and if a chunk has been read it will not be
        considered again."""
    self.logger.debug('parsing netconf v1.0')
    buf = self._session._buffer
    buf.seek(self._parsing_pos10)
    if MSG_DELIM in buf.read().decode('UTF-8'):
        buf.seek(0)
        msg, _, remaining = buf.read().decode('UTF-8').partition(MSG_DELIM)
        msg = msg.strip()
        if sys.version < '3':
            self._session._dispatch_message(msg.encode())
        else:
            self._session._dispatch_message(msg)
        self._session._buffer = StringIO()
        self._parsing_pos10 = 0
        if len(remaining.strip()) > 0:
            if type(self._session.parser) != DefaultXMLParser:
                self.logger.debug('send remaining data to SAX parser')
                self._session.parser.parse(remaining.encode())
            else:
                self.logger.debug('Trying another round of parsing since there is still data')
                self._session._buffer.write(remaining.encode())
                self._parse10()
    else:
        self._parsing_pos10 = buf.tell() - MSG_DELIM_LEN
        if self._parsing_pos10 < 0:
            self._parsing_pos10 = 0