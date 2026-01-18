import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KrbTgsReq(KrbAsReq):
    """The KRB_TGS_REQ is the same as KRB_AS_REQ but with a different MESSAGE_TYPE."""
    MESSAGE_TYPE = KerberosMessageType.tgs_req