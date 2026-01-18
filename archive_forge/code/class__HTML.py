import xml.etree.ElementTree as ET
from html import escape
class _HTML:

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError
        attr = attr.lower()
        if attr.endswith('_'):
            attr = attr[:-1]
        if '__' in attr:
            attr = attr.replace('__', ':')
        if attr == 'comment':
            return Element(ET.Comment, {})
        else:
            return Element(attr, {})

    def __call__(self, *args):
        return ElementList(args)

    def quote(self, arg):
        if arg is None:
            return ''
        return escape(arg, True)

    def str(self, arg, encoding=None):
        if isinstance(arg, str):
            return arg
        if isinstance(str, bytes):
            return arg.encode(default_encoding)
        if arg is None:
            return ''
        if isinstance(arg, (list, tuple)):
            return ''.join(map(self.str, arg))
        return str(arg)