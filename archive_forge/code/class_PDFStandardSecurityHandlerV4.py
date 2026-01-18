import itertools
import logging
import re
import struct
from hashlib import sha256, md5, sha384, sha512
from typing import (
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from . import settings
from .arcfour import Arcfour
from .data_structures import NumberTree
from .pdfparser import PDFSyntaxError, PDFParser, PDFStreamParser
from .pdftypes import (
from .psparser import PSEOF, literal_name, LIT, KWD
from .utils import choplist, decode_text, nunpack, format_int_roman, format_int_alpha
class PDFStandardSecurityHandlerV4(PDFStandardSecurityHandler):
    supported_revisions: Tuple[int, ...] = (4,)

    def init_params(self) -> None:
        super().init_params()
        self.length = 128
        self.cf = dict_value(self.param.get('CF'))
        self.stmf = literal_name(self.param['StmF'])
        self.strf = literal_name(self.param['StrF'])
        self.encrypt_metadata = bool(self.param.get('EncryptMetadata', True))
        if self.stmf != self.strf:
            error_msg = 'Unsupported crypt filter: param=%r' % self.param
            raise PDFEncryptionError(error_msg)
        self.cfm = {}
        for k, v in self.cf.items():
            f = self.get_cfm(literal_name(v['CFM']))
            if f is None:
                error_msg = 'Unknown crypt filter method: param=%r' % self.param
                raise PDFEncryptionError(error_msg)
            self.cfm[k] = f
        self.cfm['Identity'] = self.decrypt_identity
        if self.strf not in self.cfm:
            error_msg = 'Undefined crypt filter: param=%r' % self.param
            raise PDFEncryptionError(error_msg)
        return

    def get_cfm(self, name: str) -> Optional[Callable[[int, int, bytes], bytes]]:
        if name == 'V2':
            return self.decrypt_rc4
        elif name == 'AESV2':
            return self.decrypt_aes128
        else:
            return None

    def decrypt(self, objid: int, genno: int, data: bytes, attrs: Optional[Dict[str, Any]]=None, name: Optional[str]=None) -> bytes:
        if not self.encrypt_metadata and attrs is not None:
            t = attrs.get('Type')
            if t is not None and literal_name(t) == 'Metadata':
                return data
        if name is None:
            name = self.strf
        return self.cfm[name](objid, genno, data)

    def decrypt_identity(self, objid: int, genno: int, data: bytes) -> bytes:
        return data

    def decrypt_aes128(self, objid: int, genno: int, data: bytes) -> bytes:
        assert self.key is not None
        key = self.key + struct.pack('<L', objid)[:3] + struct.pack('<L', genno)[:2] + b'sAlT'
        hash = md5(key)
        key = hash.digest()[:min(len(key), 16)]
        initialization_vector = data[:16]
        ciphertext = data[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(initialization_vector), backend=default_backend())
        return cipher.decryptor().update(ciphertext)