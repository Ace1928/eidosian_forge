from io import BytesIO
from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _aiff(h, f):
    if not h.startswith(b'FORM'):
        return None
    if h[8:12] in {b'AIFC', b'AIFF'}:
        return 'x-aiff'
    else:
        return None