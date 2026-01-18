import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _CRAM_MD5_AUTH(self, challenge):
    """ Authobject to use with CRAM-MD5 authentication. """
    import hmac
    pwd = self.password.encode('utf-8') if isinstance(self.password, str) else self.password
    return self.user + ' ' + hmac.HMAC(pwd, challenge, 'md5').hexdigest()