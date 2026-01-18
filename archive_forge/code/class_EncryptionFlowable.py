import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
class EncryptionFlowable(StandardEncryption, Flowable):
    """Drop this in your Platypus story and it will set up the encryption options.

    If you do it multiple times, the last one before saving will win."""

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def draw(self):
        encryptCanvas(self.canv, self.userPassword, self.ownerPassword, self.canPrint, self.canModify, self.canCopy, self.canAnnotate)