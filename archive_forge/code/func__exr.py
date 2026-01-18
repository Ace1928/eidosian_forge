from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _exr(h):
    if h.startswith(b'v/1\x01'):
        return 'exr'