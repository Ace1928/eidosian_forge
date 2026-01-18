import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
Calculate automatic user identification.

    :returns: (realname, email), either of which may be None if they can't be
    determined.

    Only used when none is set in the environment or the id file.

    This only returns an email address if we can be fairly sure the
    address is reasonable, ie if /etc/mailname is set on unix.

    This doesn't use the FQDN as the default domain because that may be
    slow, and it doesn't use the hostname alone because that's not normally
    a reasonable address.
    