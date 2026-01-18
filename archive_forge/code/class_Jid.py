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
class Jid:

    @staticmethod
    def normalize(number):
        if '@' in number:
            return number
        elif '-' in number:
            return '%s@%s' % (number, YowConstants.WHATSAPP_GROUP_SERVER)
        return '%s@%s' % (number, YowConstants.WHATSAPP_SERVER)