import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
class NativeIO(io.BytesIO):

    def write(self, s):
        if isinstance(s, unicode):
            s = s.encode('ascii')
        super(NativeIO, self).write(s)