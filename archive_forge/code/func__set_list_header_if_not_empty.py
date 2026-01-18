import mimetypes
from email import charset as Charset
from email import encoders as Encoders
from email import generator, message_from_string
from email.errors import HeaderParseError
from email.header import Header
from email.headerregistry import Address, parser
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, getaddresses, make_msgid
from io import BytesIO, StringIO
from pathlib import Path
from django.conf import settings
from django.core.mail.utils import DNS_NAME
from django.utils.encoding import force_str, punycode
def _set_list_header_if_not_empty(self, msg, header, values):
    """
        Set msg's header, either from self.extra_headers, if present, or from
        the values argument.
        """
    if values:
        try:
            value = self.extra_headers[header]
        except KeyError:
            value = ', '.join((str(v) for v in values))
        msg[header] = value