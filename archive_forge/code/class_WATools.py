import os
from .constants import YowConstants
import codecs, sys
import logging
import tempfile
import base64
import hashlib
import os.path, mimetypes
import uuid
from consonance.structs.keypair import KeyPair
from appdirs import user_config_dir
from .optionalmodules import PILOptionalModule, FFVideoOptionalModule
class WATools:

    @staticmethod
    def generateIdentity():
        return os.urandom(20)

    @classmethod
    def generatePhoneId(cls):
        """
        :return:
        :rtype: str
        """
        return str(cls.generateUUID())

    @classmethod
    def generateDeviceId(cls):
        """
        :return:
        :rtype: bytes
        """
        return cls.generateUUID().bytes

    @classmethod
    def generateUUID(cls):
        """
        :return:
        :rtype: uuid.UUID
        """
        return uuid.uuid4()

    @classmethod
    def generateKeyPair(cls):
        """
        :return:
        :rtype: KeyPair
        """
        return KeyPair.generate()

    @staticmethod
    def getFileHashForUpload(filePath):
        sha1 = hashlib.sha256()
        f = open(filePath, 'rb')
        try:
            sha1.update(f.read())
        finally:
            f.close()
        b64Hash = base64.b64encode(sha1.digest())
        return b64Hash if type(b64Hash) is str else b64Hash.decode()