from io import BytesIO
from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _au(h, f):
    if h.startswith(b'.snd'):
        return 'basic'
    else:
        return None