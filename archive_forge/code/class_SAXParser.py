import os
from threading import Lock
import difflib
from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.builder import E
from xml.sax._exceptions import SAXParseException
from xml.sax import make_parser
from ncclient.transport.parser import DefaultXMLParser
from ncclient.operations import rpc
from ncclient.transport.parser import SAXFilterXMLNotFoundError
from ncclient.transport.parser import MSG_DELIM, MSG_DELIM_LEN
from ncclient.operations.errors import OperationError
import logging
from ncclient.xml_ import BASE_NS_1_0
class SAXParser(ContentHandler):

    def __init__(self, session):
        ContentHandler.__init__(self)
        self._currenttag = None
        self._ignoretag = None
        self._defaulttags = []
        self._session = session
        self._validate_reply_and_sax_tag = False
        self._lock = Lock()
        self.nc_namespace = None

    def startElement(self, tag, attributes):
        if tag in ['rpc-reply', 'nc:rpc-reply']:
            if tag == 'nc:rpc-reply':
                self.nc_namespace = BASE_NS_1_0
            with self._lock:
                listeners = list(self._session._listeners)
            rpc_reply_listener = [i for i in listeners if isinstance(i, rpc.RPCReplyListener)]
            rpc_msg_id = attributes._attrs['message-id']
            if rpc_msg_id in rpc_reply_listener[0]._id2rpc:
                rpc_reply_handler = rpc_reply_listener[0]._id2rpc[rpc_msg_id]
                if hasattr(rpc_reply_handler, '_filter_xml') and rpc_reply_handler._filter_xml is not None:
                    self._cur = self._root = _get_sax_parser_root(rpc_reply_handler._filter_xml)
                else:
                    raise SAXFilterXMLNotFoundError(rpc_reply_handler)
            else:
                raise OperationError("Unknown 'message-id': %s" % rpc_msg_id)
        if self._ignoretag is not None:
            return
        if self._cur == self._root and self._cur.tag == tag:
            node = self._root
        else:
            node = self._cur.find(tag, namespaces={'nc': self.nc_namespace})
        if self._validate_reply_and_sax_tag:
            if tag != self._root.tag:
                self._write_buffer(tag, format_str='<{}>\n')
                self._cur = E(tag, self._cur)
            else:
                self._write_buffer(tag, format_str='<{}{}>', **attributes)
                self._cur = node
                self._currenttag = tag
            self._validate_reply_and_sax_tag = False
            self._defaulttags.append(tag)
        elif node is not None:
            self._write_buffer(tag, format_str='<{}{}>', **attributes)
            self._cur = node
            self._currenttag = tag
        elif tag in ['rpc-reply', 'nc:rpc-reply']:
            self._write_buffer(tag, format_str='<{}{}>', **attributes)
            self._defaulttags.append(tag)
            self._validate_reply_and_sax_tag = True
        else:
            self._currenttag = None
            self._ignoretag = tag

    def endElement(self, tag):
        if self._ignoretag == tag:
            self._ignoretag = None
        if tag in self._defaulttags:
            self._write_buffer(tag, format_str='</{}>\n')
        elif self._cur.tag == tag:
            self._write_buffer(tag, format_str='</{}>\n')
            self._cur = self._cur.getparent()
        self._currenttag = None

    def characters(self, content):
        if self._currenttag is not None:
            self._write_buffer(content, format_str='{}')

    def _write_buffer(self, content, format_str, **kwargs):
        self._session._buffer.seek(0, os.SEEK_END)
        attrs = ''
        for name, value in kwargs.items():
            attr = ' {}={}'.format(name, quoteattr(value))
            attrs = attrs + attr
        data = format_str.format(escape(content), attrs)
        self._session._buffer.write(str.encode(data))