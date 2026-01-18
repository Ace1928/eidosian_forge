from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def _MonkeyPatchHttpForPython_3x():
    import http
    old_begin = http.client.HTTPResponse.begin

    def PatchedBegin(self):
        old_begin(self)
        if self.debuglevel > 0:
            for hdr, val in self.headers.items():
                print('header:', hdr + ':', val)
    http.client.HTTPResponse.begin = PatchedBegin