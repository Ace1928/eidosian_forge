from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class ContentDispositionHeader(ParameterizedMIMEHeader):
    value_parser = staticmethod(parser.parse_content_disposition_header)

    def init(self, *args, **kw):
        super().init(*args, **kw)
        cd = self._parse_tree.content_disposition
        self._content_disposition = cd if cd is None else utils._sanitize(cd)

    @property
    def content_disposition(self):
        return self._content_disposition