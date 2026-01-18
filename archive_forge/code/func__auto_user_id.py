import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _auto_user_id():
    """Calculate automatic user identification.

    :returns: (realname, email), either of which may be None if they can't be
    determined.

    Only used when none is set in the environment or the id file.

    This only returns an email address if we can be fairly sure the
    address is reasonable, ie if /etc/mailname is set on unix.

    This doesn't use the FQDN as the default domain because that may be
    slow, and it doesn't use the hostname alone because that's not normally
    a reasonable address.
    """
    if sys.platform == 'win32':
        return (None, None)
    default_mail_domain = _get_default_mail_domain()
    if not default_mail_domain:
        return (None, None)
    import pwd
    uid = os.getuid()
    try:
        w = pwd.getpwuid(uid)
    except KeyError:
        trace.mutter('no passwd entry for uid %d?' % uid)
        return (None, None)
    gecos = w.pw_gecos
    if isinstance(gecos, bytes):
        try:
            gecos = gecos.decode('utf-8')
            encoding = 'utf-8'
        except UnicodeError:
            try:
                encoding = osutils.get_user_encoding()
                gecos = gecos.decode(encoding)
            except UnicodeError:
                trace.mutter('cannot decode passwd entry %s' % w)
                return (None, None)
    username = w.pw_name
    if isinstance(username, bytes):
        try:
            username = username.decode(encoding)
        except UnicodeError:
            trace.mutter('cannot decode passwd entry %s' % w)
            return (None, None)
    comma = gecos.find(',')
    if comma == -1:
        realname = gecos
    else:
        realname = gecos[:comma]
    return (realname, username + '@' + default_mail_domain)