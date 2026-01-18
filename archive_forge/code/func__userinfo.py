from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
@property
def _userinfo(self):
    netloc = self.netloc
    userinfo, have_info, hostinfo = netloc.rpartition(b'@')
    if have_info:
        username, have_password, password = userinfo.partition(b':')
        if not have_password:
            password = None
    else:
        username = password = None
    return (username, password)