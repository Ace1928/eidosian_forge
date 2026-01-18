import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosPrincipalNameType(enum.IntEnum):
    unknown = 0
    principal = 1
    srv_inst = 2
    srv_hst = 3
    srv_xhst = 4
    uid = 5
    x500_principal = 6
    smtp_name = 7
    enterprise = 10

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosPrincipalNameType', str]:
        return {KerberosPrincipalNameType.unknown: 'NT-UNKNOWN', KerberosPrincipalNameType.principal: 'NT-PRINCIPAL', KerberosPrincipalNameType.srv_inst: 'NT-SRV-INST', KerberosPrincipalNameType.srv_hst: 'NT-SRV-HST', KerberosPrincipalNameType.srv_xhst: 'NT-SRV-XHST', KerberosPrincipalNameType.uid: 'NT-UID', KerberosPrincipalNameType.x500_principal: 'NT-X500-PRINCIPAL', KerberosPrincipalNameType.smtp_name: 'NT-SMTP-NAME', KerberosPrincipalNameType.enterprise: 'NT-ENTERPRISE'}