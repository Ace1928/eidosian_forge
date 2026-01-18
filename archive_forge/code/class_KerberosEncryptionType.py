import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosEncryptionType(enum.IntEnum):
    des_cbc_crc = 1
    des_cbc_md4 = 2
    des_cbc_md5 = 3
    des_cbc_raw = 4
    des3_cbc_raw = 6
    des3_cbc_sha1 = 16
    aes128_cts_hmac_sha1_96 = 17
    aes256_cts_hmac_sha1_96 = 18
    aes128_cts_hmac_sha256_128 = 19
    aes256_cts_hmac_sha384_192 = 20
    rc4_hmac = 23
    rc4_hmac_exp = 24
    camellia128_cts_cmac = 25
    camellia256_cts_cmac = 26

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosEncryptionType', str]:
        return {KerberosEncryptionType.des_cbc_crc: 'DES_CBC_CRC', KerberosEncryptionType.des_cbc_md4: 'DES_CBC_MD4', KerberosEncryptionType.des_cbc_md5: 'DES_CBC_MD5', KerberosEncryptionType.des_cbc_raw: 'DES_CBC_RAW', KerberosEncryptionType.des3_cbc_raw: 'DES3_CBC_RAW', KerberosEncryptionType.des3_cbc_sha1: 'DES3_CBC_SHA1', KerberosEncryptionType.aes128_cts_hmac_sha1_96: 'AES128_CTS_HMAC_SHA1_96', KerberosEncryptionType.aes256_cts_hmac_sha1_96: 'AES256_CTS_HMAC_SHA1_96', KerberosEncryptionType.aes128_cts_hmac_sha256_128: 'AES128_CTS_HMAC_SHA256_128', KerberosEncryptionType.aes256_cts_hmac_sha384_192: 'AES256_CTS_HMAC_SHA384_192', KerberosEncryptionType.rc4_hmac: 'RC4_HMAC', KerberosEncryptionType.rc4_hmac_exp: 'RC4_HMAC_EXP', KerberosEncryptionType.camellia128_cts_cmac: 'CAMELLIA128_CTS_CMAC', KerberosEncryptionType.camellia256_cts_cmac: 'CAMELLIA256_CTS_CMAC'}