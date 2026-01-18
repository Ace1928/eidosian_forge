import base64
from datetime import datetime
import hashlib
from gslib.utils.constants import UTF8
import six
from six.moves import urllib
def GetFinalUrl(raw_signature, host, path, canonical_query_string):
    """Get the final signed url."""
    signature = base64.b16encode(raw_signature).lower().decode()
    return _SIGNED_URL_FORMAT.format(host=host, path=path, sig=signature, query_string=canonical_query_string)