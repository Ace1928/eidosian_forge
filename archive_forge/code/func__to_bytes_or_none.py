import logging
from email import encoders as Encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from io import BytesIO
from twisted import version as twisted_version
from twisted.internet import defer, ssl
from twisted.python.versions import Version
from scrapy.utils.misc import arg_to_iter
from scrapy.utils.python import to_bytes
def _to_bytes_or_none(text):
    if text is None:
        return None
    return to_bytes(text)