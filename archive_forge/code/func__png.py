from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _png(h):
    if h.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'