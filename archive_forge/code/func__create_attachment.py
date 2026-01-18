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
def _create_attachment(self, filename, content, mimetype=None):
    """
        Convert the filename, content, mimetype triple into a MIME attachment
        object.
        """
    attachment = self._create_mime_attachment(content, mimetype)
    if filename:
        try:
            filename.encode('ascii')
        except UnicodeEncodeError:
            filename = ('utf-8', '', filename)
        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
    return attachment