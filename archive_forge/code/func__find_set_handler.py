import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
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