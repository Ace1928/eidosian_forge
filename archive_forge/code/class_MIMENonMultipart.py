from email import errors
from email.mime.base import MIMEBase
class MIMENonMultipart(MIMEBase):
    """Base class for MIME non-multipart type messages."""

    def attach(self, payload):
        raise errors.MultipartConversionError('Cannot attach additional subparts to non-multipart/*')