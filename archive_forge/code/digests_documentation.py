import hashlib
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_native_str, to_bytes, render_bytes, consteq
from passlib.utils.compat import unicode, str_to_uascii
import passlib.utils.handlers as uh
from passlib.crypto.digest import lookup_hash
normalize hash to native string, and validate it