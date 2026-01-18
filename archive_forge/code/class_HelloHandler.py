import logging
from threading import Thread, Lock, Event
import ncclient.transport
from ncclient.xml_ import *
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport.errors import TransportError, SessionError, SessionCloseError
from ncclient.transport.notify import Notification
class HelloHandler(SessionListener):

    def __init__(self, init_cb, error_cb):
        self._init_cb = init_cb
        self._error_cb = error_cb

    def callback(self, root, raw):
        tag, attrs = root
        if tag == qualify('hello') or tag == 'hello':
            try:
                id, capabilities = HelloHandler.parse(raw)
            except Exception as e:
                self._error_cb(e)
            else:
                self._init_cb(id, capabilities)

    def errback(self, err):
        self._error_cb(err)

    @staticmethod
    def build(capabilities, device_handler):
        """Given a list of capability URI's returns <hello> message XML string"""
        if device_handler:
            xml_namespace_kwargs = {'nsmap': device_handler.get_xml_base_namespace_dict()}
        else:
            xml_namespace_kwargs = {}
        hello = new_ele('hello', **xml_namespace_kwargs)
        caps = sub_ele(hello, 'capabilities')

        def fun(uri):
            sub_ele(caps, 'capability').text = uri
        if sys.version < '3':
            map(fun, capabilities)
        else:
            list(map(fun, capabilities))
        return to_xml(hello)

    @staticmethod
    def parse(raw):
        """Returns tuple of (session-id (str), capabilities (Capabilities)"""
        sid, capabilities = (0, [])
        root = to_ele(raw)
        for child in root.getchildren():
            if child.tag == qualify('session-id') or child.tag == 'session-id':
                sid = child.text
            elif child.tag == qualify('capabilities') or child.tag == 'capabilities':
                for cap in child.getchildren():
                    if cap.tag == qualify('capability') or cap.tag == 'capability':
                        capabilities.append(cap.text)
        return (sid, Capabilities(capabilities))