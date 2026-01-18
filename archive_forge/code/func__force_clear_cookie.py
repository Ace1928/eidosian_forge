from __future__ import annotations
import binascii
import datetime
import json
import os
import re
import sys
import typing as t
import uuid
from dataclasses import asdict, dataclass
from http.cookies import Morsel
from tornado import escape, httputil, web
from traitlets import Bool, Dict, Type, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_server.transutils import _i18n
from .security import passwd_check, set_password
from .utils import get_anonymous_username
def _force_clear_cookie(self, handler: web.RequestHandler, name: str, path: str='/', domain: str | None=None) -> None:
    """Deletes the cookie with the given name.

        Tornado's cookie handling currently (Jan 2018) stores cookies in a dict
        keyed by name, so it can only modify one cookie with a given name per
        response. The browser can store multiple cookies with the same name
        but different domains and/or paths. This method lets us clear multiple
        cookies with the same name.

        Due to limitations of the cookie protocol, you must pass the same
        path and domain to clear a cookie as were used when that cookie
        was set (but there is no way to find out on the server side
        which values were used for a given cookie).
        """
    name = escape.native_str(name)
    expires = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=365)
    morsel: Morsel[t.Any] = Morsel()
    morsel.set(name, '', '""')
    morsel['expires'] = httputil.format_timestamp(expires)
    morsel['path'] = path
    if domain:
        morsel['domain'] = domain
    handler.add_header('Set-Cookie', morsel.OutputString())