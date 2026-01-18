import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
class CookiePolicy:
    """Defines which cookies get accepted from and returned to server.

    May also modify cookies, though this is probably a bad idea.

    The subclass DefaultCookiePolicy defines the standard rules for Netscape
    and RFC 2965 cookies -- override that if you want a customized policy.

    """

    def set_ok(self, cookie, request):
        """Return true if (and only if) cookie should be accepted from server.

        Currently, pre-expired cookies never get this far -- the CookieJar
        class deletes such cookies itself.

        """
        raise NotImplementedError()

    def return_ok(self, cookie, request):
        """Return true if (and only if) cookie should be returned to server."""
        raise NotImplementedError()

    def domain_return_ok(self, domain, request):
        """Return false if cookies should not be returned, given cookie domain.
        """
        return True

    def path_return_ok(self, path, request):
        """Return false if cookies should not be returned, given cookie path.
        """
        return True