from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _rast(h):
    """Sun raster file"""
    if h.startswith(b'Y\xa6j\x95'):
        return 'rast'