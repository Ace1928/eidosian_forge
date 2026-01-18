from typing import Dict
import dns.enum
import dns.exception
class RdataType(dns.enum.IntEnum):
    """DNS Rdata Type"""
    TYPE0 = 0
    NONE = 0
    A = 1
    NS = 2
    MD = 3
    MF = 4
    CNAME = 5
    SOA = 6
    MB = 7
    MG = 8
    MR = 9
    NULL = 10
    WKS = 11
    PTR = 12
    HINFO = 13
    MINFO = 14
    MX = 15
    TXT = 16
    RP = 17
    AFSDB = 18
    X25 = 19
    ISDN = 20
    RT = 21
    NSAP = 22
    NSAP_PTR = 23
    SIG = 24
    KEY = 25
    PX = 26
    GPOS = 27
    AAAA = 28
    LOC = 29
    NXT = 30
    SRV = 33
    NAPTR = 35
    KX = 36
    CERT = 37
    A6 = 38
    DNAME = 39
    OPT = 41
    APL = 42
    DS = 43
    SSHFP = 44
    IPSECKEY = 45
    RRSIG = 46
    NSEC = 47
    DNSKEY = 48
    DHCID = 49
    NSEC3 = 50
    NSEC3PARAM = 51
    TLSA = 52
    SMIMEA = 53
    HIP = 55
    NINFO = 56
    CDS = 59
    CDNSKEY = 60
    OPENPGPKEY = 61
    CSYNC = 62
    ZONEMD = 63
    SVCB = 64
    HTTPS = 65
    SPF = 99
    UNSPEC = 103
    NID = 104
    L32 = 105
    L64 = 106
    LP = 107
    EUI48 = 108
    EUI64 = 109
    TKEY = 249
    TSIG = 250
    IXFR = 251
    AXFR = 252
    MAILB = 253
    MAILA = 254
    ANY = 255
    URI = 256
    CAA = 257
    AVC = 258
    AMTRELAY = 260
    TA = 32768
    DLV = 32769

    @classmethod
    def _maximum(cls):
        return 65535

    @classmethod
    def _short_name(cls):
        return 'type'

    @classmethod
    def _prefix(cls):
        return 'TYPE'

    @classmethod
    def _extra_from_text(cls, text):
        if text.find('-') >= 0:
            try:
                return cls[text.replace('-', '_')]
            except KeyError:
                pass
        return _registered_by_text.get(text)

    @classmethod
    def _extra_to_text(cls, value, current_text):
        if current_text is None:
            return _registered_by_value.get(value)
        if current_text.find('_') >= 0:
            return current_text.replace('_', '-')
        return current_text

    @classmethod
    def _unknown_exception_class(cls):
        return UnknownRdatatype