import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
Hash PKI tokens.

    return: for asn1 or pkiz tokens, returns the hash of the passed in token
            otherwise, returns what it was passed in.
    