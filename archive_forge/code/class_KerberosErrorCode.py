import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosErrorCode(enum.IntEnum):
    none = 0
    name_exp = 1
    service_exp = 2
    bad_pvno = 3
    c_old_mast_kvno = 4
    s_old_mast_kvno = 5
    c_principal_unknown = 6
    s_principal_unknown = 7
    principal_not_unique = 8
    null_key = 9
    cannot_postdate = 10
    never_valid = 11
    policy = 12
    badoption = 13
    etype_nosupp = 14
    sumtype_nosupp = 15
    padata_type_nosupp = 16
    trtype_nosupp = 17
    client_revoked = 18
    service_revoked = 19
    tgt_revoked = 20
    client_notyet = 21
    service_notyet = 22
    key_expired = 23
    preauth_failed = 24
    preauth_required = 25
    server_nomatch = 26
    must_use_user2user = 27
    path_not_accepted = 28
    kdc_svc_unavailable = 29
    ap_bad_integrity = 31
    ap_txt_expired = 32
    ap_tkt_nyv = 33
    ap_repeat = 34
    ap_not_use = 35
    ap_badmatch = 36
    ap_skew = 37
    ap_badaddr = 38
    ap_badversion = 39
    ap_msg_type = 40
    ap_modified = 41
    ap_badorder = 42
    ap_badkeyver = 44
    ap_nokey = 45
    ap_mut_fail = 46
    ap_baddirection = 47
    ap_method = 48
    ap_badseq = 49
    ap_inapp_cksum = 50
    ap_path_not_accepted = 51
    response_too_big = 52
    generic = 60
    field_toolong = 61
    kdc_client_not_trusted = 62
    kdc_not_trusted = 53
    kdc_invalid_sig = 64
    kdc_key_too_weak = 65
    kdc_certificate_mismatch = 66
    ap_no_tgt = 67
    kdc_wrong_realm = 68
    ap_user_to_user_required = 69
    kdc_cant_verify_certificate = 70
    kdc_invalid_certificate = 71
    kdc_revoked_certificate = 72
    kdc_revocation_status_unknown = 73
    kdc_revocation_status_unavailable = 74
    kdc_client_name_mismatch = 75
    kdc_name_mismatch = 76

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosErrorCode', str]:
        return {KerberosErrorCode.none: 'KDC_ERR_NONE', KerberosErrorCode.name_exp: 'KDC_ERR_NAME_EXP', KerberosErrorCode.service_exp: 'KDC_ERR_SERVICE_EXP', KerberosErrorCode.bad_pvno: 'KDC_ERR_BAD_PVNO', KerberosErrorCode.c_old_mast_kvno: 'KDC_ERR_C_OLD_MAST_KVNO', KerberosErrorCode.s_old_mast_kvno: 'KDC_ERR_S_OLD_MAST_KVNO', KerberosErrorCode.c_principal_unknown: 'KDC_ERR_C_PRINCIPAL_UNKNOWN', KerberosErrorCode.s_principal_unknown: 'KDC_ERR_S_PRINCIPAL_UNKNOWN', KerberosErrorCode.principal_not_unique: 'KDC_ERR_PRINCIPAL_NOT_UNIQUE', KerberosErrorCode.null_key: 'KDC_ERR_NULL_KEY', KerberosErrorCode.cannot_postdate: 'KDC_ERR_CANNOT_POSTDATE', KerberosErrorCode.never_valid: 'KDC_ERR_NEVER_VALID', KerberosErrorCode.policy: 'KDC_ERR_POLICY', KerberosErrorCode.badoption: 'KDC_ERR_BADOPTION', KerberosErrorCode.etype_nosupp: 'KDC_ERR_ETYPE_NOSUPP', KerberosErrorCode.sumtype_nosupp: 'KDC_ERR_SUMTYPE_NOSUPP', KerberosErrorCode.padata_type_nosupp: 'KDC_ERR_PADATA_TYPE_NOSUPP', KerberosErrorCode.trtype_nosupp: 'KDC_ERR_TRTYPE_NOSUPP', KerberosErrorCode.client_revoked: 'KDC_ERR_CLIENT_REVOKED', KerberosErrorCode.service_revoked: 'KDC_ERR_SERVICE_REVOKED', KerberosErrorCode.tgt_revoked: 'KDC_ERR_TGT_REVOKED', KerberosErrorCode.client_notyet: 'KDC_ERR_CLIENT_NOTYET', KerberosErrorCode.service_notyet: 'KDC_ERR_SERVICE_NOTYET', KerberosErrorCode.key_expired: 'KDC_ERR_KEY_EXPIRED', KerberosErrorCode.preauth_failed: 'KDC_ERR_PREAUTH_FAILED', KerberosErrorCode.preauth_required: 'KDC_ERR_PREAUTH_REQUIRED', KerberosErrorCode.server_nomatch: 'KDC_ERR_SERVER_NOMATCH', KerberosErrorCode.must_use_user2user: 'KDC_ERR_MUST_USE_USER2USER', KerberosErrorCode.path_not_accepted: 'KDC_ERR_PATH_NOT_ACCEPTED', KerberosErrorCode.kdc_svc_unavailable: 'KDC_ERR_SVC_UNAVAILABLE', KerberosErrorCode.ap_bad_integrity: 'KRB_AP_ERR_BAD_INTEGRITY', KerberosErrorCode.ap_txt_expired: 'KRB_AP_ERR_TKT_EXPIRED', KerberosErrorCode.ap_tkt_nyv: 'KRB_AP_ERR_TKT_NYV', KerberosErrorCode.ap_repeat: 'KRB_AP_ERR_REPEAT', KerberosErrorCode.ap_not_use: 'KRB_AP_ERR_NOT_US', KerberosErrorCode.ap_badmatch: 'KRB_AP_ERR_BADMATCH', KerberosErrorCode.ap_skew: 'KRB_AP_ERR_SKEW', KerberosErrorCode.ap_badaddr: 'KRB_AP_ERR_BADADDR', KerberosErrorCode.ap_badversion: 'KRB_AP_ERR_BADVERSION', KerberosErrorCode.ap_msg_type: 'KRB_AP_ERR_MSG_TYPE', KerberosErrorCode.ap_modified: 'KRB_AP_ERR_MODIFIED', KerberosErrorCode.ap_badorder: 'KRB_AP_ERR_BADORDER', KerberosErrorCode.ap_badkeyver: 'KRB_AP_ERR_BADKEYVER', KerberosErrorCode.ap_nokey: 'KRB_AP_ERR_NOKEY', KerberosErrorCode.ap_mut_fail: 'KRB_AP_ERR_MUT_FAIL', KerberosErrorCode.ap_baddirection: 'KRB_AP_ERR_BADDIRECTION', KerberosErrorCode.ap_method: 'KRB_AP_ERR_METHOD', KerberosErrorCode.ap_badseq: 'KRB_AP_ERR_BADSEQ', KerberosErrorCode.ap_inapp_cksum: 'KRB_AP_ERR_INAPP_CKSUM', KerberosErrorCode.ap_path_not_accepted: 'KRB_AP_PATH_NOT_ACCEPTED', KerberosErrorCode.response_too_big: 'KRB_ERR_RESPONSE_TOO_BIG', KerberosErrorCode.generic: 'KRB_ERR_GENERIC', KerberosErrorCode.field_toolong: 'KRB_ERR_FIELD_TOOLONG', KerberosErrorCode.kdc_client_not_trusted: 'KDC_ERROR_CLIENT_NOT_TRUSTED', KerberosErrorCode.kdc_not_trusted: 'KDC_ERROR_KDC_NOT_TRUSTED', KerberosErrorCode.kdc_invalid_sig: 'KDC_ERROR_INVALID_SIG', KerberosErrorCode.kdc_key_too_weak: 'KDC_ERR_KEY_TOO_WEAK', KerberosErrorCode.kdc_certificate_mismatch: 'KDC_ERR_CERTIFICATE_MISMATCH', KerberosErrorCode.ap_no_tgt: 'KRB_AP_ERR_NO_TGT', KerberosErrorCode.kdc_wrong_realm: 'KDC_ERR_WRONG_REALM', KerberosErrorCode.ap_user_to_user_required: 'KRB_AP_ERR_USER_TO_USER_REQUIRED', KerberosErrorCode.kdc_cant_verify_certificate: 'KDC_ERR_CANT_VERIFY_CERTIFICATE', KerberosErrorCode.kdc_invalid_certificate: 'KDC_ERR_INVALID_CERTIFICATE', KerberosErrorCode.kdc_revoked_certificate: 'KDC_ERR_REVOKED_CERTIFICATE', KerberosErrorCode.kdc_revocation_status_unknown: 'KDC_ERR_REVOCATION_STATUS_UNKNOWN', KerberosErrorCode.kdc_revocation_status_unavailable: 'KDC_ERR_REVOCATION_STATUS_UNAVAILABLE', KerberosErrorCode.kdc_client_name_mismatch: 'KDC_ERR_CLIENT_NAME_MISMATCH', KerberosErrorCode.kdc_name_mismatch: 'KDC_ERR_KDC_NAME_MISMATCH'}