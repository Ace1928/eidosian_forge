from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def ack_type(capabilities):
    """Extract the ack type from a capabilities list."""
    if b'multi_ack_detailed' in capabilities:
        return MULTI_ACK_DETAILED
    elif b'multi_ack' in capabilities:
        return MULTI_ACK
    return SINGLE_ACK