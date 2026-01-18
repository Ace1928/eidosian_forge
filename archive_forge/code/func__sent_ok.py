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
def _sent_ok(self, result, to, cc, subject, nattachs):
    logger.info('Mail sent OK: To=%(mailto)s Cc=%(mailcc)s Subject="%(mailsubject)s" Attachs=%(mailattachs)d', {'mailto': to, 'mailcc': cc, 'mailsubject': subject, 'mailattachs': nattachs})