import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
class StandardEncryption:
    prepared = 0

    def __init__(self, userPassword, ownerPassword=None, canPrint=1, canModify=1, canCopy=1, canAnnotate=1, strength=None):
        """
        This class defines the encryption properties to be used while creating a pdf document.
        Once initiated, a StandardEncryption object can be applied to a Canvas or a BaseDocTemplate.
        The userPassword parameter sets the user password on the encrypted pdf.
        The ownerPassword parameter sets the owner password on the encrypted pdf.
        The boolean flags canPrint, canModify, canCopy, canAnnotate determine wether a user can
        perform the corresponding actions on the pdf when only a user password has been supplied.
        If the user supplies the owner password while opening the pdf, all actions can be performed regardless
        of the flags.
        Note that the security provided by these encryption settings (and even more so for the flags) is very weak.
        """
        self.userPassword = userPassword
        if ownerPassword:
            self.ownerPassword = ownerPassword
        else:
            self.ownerPassword = userPassword
        if strength is None:
            strength = rl_config.encryptionStrength
        if strength == 40:
            self.revision = 2
        elif strength == 128:
            self.revision = 3
        elif strength == 256:
            if not pyaes:
                raise ValueError('strength==256 is not supported as package pyaes is not importable')
            self.revision = 5
        else:
            raise ValueError('Unknown encryption strength=%s' % repr(strength))
        self.canPrint = canPrint
        self.canModify = canModify
        self.canCopy = canCopy
        self.canAnnotate = canAnnotate
        self.O = self.U = self.P = self.key = self.OE = self.UE = self.Perms = None

    def setAllPermissions(self, value):
        self.canPrint = self.canModify = self.canCopy = self.canAnnotate = value

    def permissionBits(self):
        p = 0
        if self.canPrint:
            p = p | printable
        if self.canModify:
            p = p | modifiable
        if self.canCopy:
            p = p | copypastable
        if self.canAnnotate:
            p = p | annotatable
        p = p | higherbits
        return p

    def encode(self, t):
        """encode a string, stream, text"""
        if not self.prepared:
            raise ValueError('encryption not prepared!')
        if self.objnum is None:
            raise ValueError('not registered in PDF object')
        return encodePDF(self.key, self.objnum, self.version, t, revision=self.revision)

    def prepare(self, document, overrideID=None):
        if DEBUG:
            print('StandardEncryption.prepare(...) - revision %d' % self.revision)
        if self.prepared:
            raise ValueError('encryption already prepared!')
        if overrideID:
            internalID = overrideID
        else:
            externalID = document.ID()
            internalID = document.signature.digest()
            if CLOBBERID:
                internalID = 'xxxxxxxxxxxxxxxx'
        if DEBUG:
            print('userPassword    = %r' % self.userPassword)
            print('ownerPassword   = %r' % self.ownerPassword)
            print('internalID      = %r' % internalID)
        self.P = int(self.permissionBits() - 2 ** 31)
        if CLOBBERPERMISSIONS:
            self.P = -44
        if DEBUG:
            print('self.P          = %s' % repr(self.P))
        if self.revision == 5:
            iv = b'\x00' * 16
            uvs = os_urandom(8)
            uks = os_urandom(8)
            self.key = asBytes(os_urandom(32))
            if DEBUG:
                print('uvs      (hex)  = %s' % hexText(uvs))
                print('uks      (hex)  = %s' % hexText(uks))
                print('self.key (hex)  = %s' % hexText(self.key))
            md = sha256(asBytes(self.userPassword[:127]) + uvs)
            self.U = md.digest() + uvs + uks
            if DEBUG:
                print('self.U (hex)  = %s' % hexText(self.U))
            md = sha256(asBytes(self.userPassword[:127]) + uks)
            encrypter = pyaes.Encrypter(pyaes.AESModeOfOperationCBC(md.digest(), iv=iv))
            self.UE = encrypter.feed(self.key)
            self.UE += encrypter.feed()
            if DEBUG:
                print('self.UE (hex)  = %s' % hexText(self.UE))
            ovs = os_urandom(8)
            oks = os_urandom(8)
            md = sha256(asBytes(self.ownerPassword[:127]) + ovs + self.U)
            self.O = md.digest() + ovs + oks
            if DEBUG:
                print('self.O (hex)  = %s' % hexText(self.O))
            md = sha256(asBytes(self.ownerPassword[:127]) + oks + self.U)
            encrypter = pyaes.Encrypter(pyaes.AESModeOfOperationCBC(md.digest(), iv=iv))
            self.OE = encrypter.feed(self.key)
            self.OE += encrypter.feed()
            if DEBUG:
                print('self.OE (hex)  = %s' % hexText(self.OE))
            permsarr = [self.P & 255, self.P >> 8 & 255, self.P >> 16 & 255, self.P >> 24 & 255, 255, 255, 255, 255, ord('T'), ord('a'), ord('d'), ord('b'), 1, 1, 1, 1]
            encrypter = pyaes.Encrypter(pyaes.AESModeOfOperationCBC(self.key, iv=iv))
            self.Perms = encrypter.feed(bytes(permsarr))
            self.Perms += encrypter.feed()
            if DEBUG:
                print('self.Perms (hex)  = %s' % hexText(self.Perms))
        elif self.revision in (2, 3):
            self.O = computeO(self.userPassword, self.ownerPassword, self.revision)
            if DEBUG:
                print('self.O (as hex) = %s' % hexText(self.O))
            self.key = encryptionkey(self.userPassword, self.O, self.P, internalID, revision=self.revision)
            if DEBUG:
                print('self.key (hex)  = %s' % hexText(self.key))
            self.U = computeU(self.key, revision=self.revision, documentId=internalID)
            if DEBUG:
                print('self.U (as hex) = %s' % hexText(self.U))
        self.objnum = self.version = None
        self.prepared = 1

    def register(self, objnum, version):
        if not self.prepared:
            raise ValueError('encryption not prepared!')
        self.objnum = objnum
        self.version = version

    def info(self):
        if not self.prepared:
            raise ValueError('encryption not prepared!')
        return StandardEncryptionDictionary(O=self.O, OE=self.OE, U=self.U, UE=self.UE, P=self.P, Perms=self.Perms, revision=self.revision)