import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
class ContentManager:

    def __init__(self):
        self.get_handlers = {}
        self.set_handlers = {}

    def add_get_handler(self, key, handler):
        self.get_handlers[key] = handler

    def get_content(self, msg, *args, **kw):
        content_type = msg.get_content_type()
        if content_type in self.get_handlers:
            return self.get_handlers[content_type](msg, *args, **kw)
        maintype = msg.get_content_maintype()
        if maintype in self.get_handlers:
            return self.get_handlers[maintype](msg, *args, **kw)
        if '' in self.get_handlers:
            return self.get_handlers[''](msg, *args, **kw)
        raise KeyError(content_type)

    def add_set_handler(self, typekey, handler):
        self.set_handlers[typekey] = handler

    def set_content(self, msg, obj, *args, **kw):
        if msg.get_content_maintype() == 'multipart':
            raise TypeError('set_content not valid on multipart')
        handler = self._find_set_handler(msg, obj)
        msg.clear_content()
        handler(msg, obj, *args, **kw)

    def _find_set_handler(self, msg, obj):
        full_path_for_error = None
        for typ in type(obj).__mro__:
            if typ in self.set_handlers:
                return self.set_handlers[typ]
            qname = typ.__qualname__
            modname = getattr(typ, '__module__', '')
            full_path = '.'.join((modname, qname)) if modname else qname
            if full_path_for_error is None:
                full_path_for_error = full_path
            if full_path in self.set_handlers:
                return self.set_handlers[full_path]
            if qname in self.set_handlers:
                return self.set_handlers[qname]
            name = typ.__name__
            if name in self.set_handlers:
                return self.set_handlers[name]
        if None in self.set_handlers:
            return self.set_handlers[None]
        raise KeyError(full_path_for_error)