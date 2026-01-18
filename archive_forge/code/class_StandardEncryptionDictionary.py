import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
class StandardEncryptionDictionary(PDFObject):
    __RefOnly__ = 1

    def __init__(self, O, OE, U, UE, P, Perms, revision):
        self.O, self.OE, self.U, self.UE, self.P, self.Perms = (O, OE, U, UE, P, Perms)
        self.revision = revision

    def format(self, document):
        from reportlab.pdfbase.pdfdoc import DummyDoc, PDFDictionary, PDFName
        dummy = DummyDoc()
        dict = {'Filter': PDFName('Standard'), 'O': hexText(self.O), 'U': hexText(self.U), 'P': self.P}
        if self.revision == 5:
            dict['Length'] = 256
            dict['R'] = 5
            dict['V'] = 5
            dict['O'] = hexText(self.O)
            dict['U'] = hexText(self.U)
            dict['OE'] = hexText(self.OE)
            dict['UE'] = hexText(self.UE)
            dict['Perms'] = hexText(self.Perms)
            dict['StrF'] = PDFName('StdCF')
            dict['StmF'] = PDFName('StdCF')
            stdcf = {'Length': 32, 'AuthEvent': PDFName('DocOpen'), 'CFM': PDFName('AESV3')}
            cf = {'StdCF': PDFDictionary(stdcf)}
            dict['CF'] = PDFDictionary(cf)
        elif self.revision == 3:
            dict['Length'] = 128
            dict['R'] = 3
            dict['V'] = 2
        else:
            dict['R'] = 2
            dict['V'] = 1
        pdfdict = PDFDictionary(dict)
        return pdfdict.format(dummy)