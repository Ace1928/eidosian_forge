import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosPADataType(enum.IntEnum):
    tgs_req = 1
    enc_timestamp = 2
    pw_salt = 3
    reserved = 4
    enc_unix_time = 5
    sandia_secureid = 6
    sesame = 7
    osf_dce = 8
    cybersafe_secureid = 9
    afs3_salt = 10
    etype_info = 11
    sam_challenge = 12
    sam_response = 13
    pk_as_req_old = 14
    pk_as_rep_old = 15
    pk_as_req = 16
    pk_as_rep = 17
    pk_ocsp_response = 18
    etype_info2 = 19
    use_specified_kvno = 20
    svr_referral_info = 20
    sam_redirect = 21
    get_from_typed_data = 22
    td_padata = 22
    sam_etype_info = 23
    alt_princ = 24
    server_referral = 25
    sam_challenge2 = 30
    sam_response2 = 31
    extra_tgt = 41
    td_pkinit_cms_certificates = 101
    td_krb_principal = 102
    td_krb_realm = 103
    td_trusted_certifiers = 104
    td_certificate_index = 105
    td_app_defined_error = 106
    td_req_nonce = 107
    td_req_seq = 108
    td_dh_parameters = 109
    td_cms_digest_algorithms = 111
    td_cert_digest_algorithms = 112
    pac_request = 128
    for_user = 128
    for_x509_user = 130
    for_check_dups = 131
    as_checksum = 132
    fx_cookie = 133
    authentication_set = 134
    auth_set_selected = 165
    fx_fast = 136
    fx_error = 137
    encrypted_challenge = 138
    otp_challenge = 141
    otp_request = 142
    otp_confirm = 143
    otp_pin_change = 144
    epak_as_req = 145
    epak_as_rep = 146
    pkinit_kx = 147
    pku2u_name = 148
    enc_pa_rep = 149
    as_freshness = 150
    spake = 151
    kerb_key_list_req = 161
    kerb_key_list_rep = 162
    supported_etypes = 165
    extended_error = 166
    pac_options = 167

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosPADataType', str]:
        return {KerberosPADataType.tgs_req: 'PA-TGS-REQ', KerberosPADataType.enc_timestamp: 'PA-ENC-TIMESTAMP', KerberosPADataType.pw_salt: 'PA-PW-SALT', KerberosPADataType.reserved: 'reserved', KerberosPADataType.enc_unix_time: 'PA-ENC-UNIX-TIME', KerberosPADataType.sandia_secureid: 'PA-SANDIA-SECUREID', KerberosPADataType.sesame: 'PA-SESAME', KerberosPADataType.osf_dce: 'PA-OSF-DCE', KerberosPADataType.cybersafe_secureid: 'PA-CYBERSAFE-SECUREID', KerberosPADataType.afs3_salt: 'PA-AFS3-SALT', KerberosPADataType.etype_info: 'PA-ETYPE-INFO', KerberosPADataType.sam_challenge: 'PA-SAM-CHALLENGE', KerberosPADataType.sam_response: 'PA-SAM-RESPONSE', KerberosPADataType.pk_as_req_old: 'PA-PK-AS-REQ_OLD', KerberosPADataType.pk_as_rep_old: 'PA-PK-AS-REP_OLD', KerberosPADataType.pk_as_req: 'PA-PK-AS-REQ', KerberosPADataType.pk_as_rep: 'PA-PK-AS-REP', KerberosPADataType.pk_ocsp_response: 'PA-PK-OCSP-RESPONSE', KerberosPADataType.etype_info2: 'PA-ETYPE-INFO2', KerberosPADataType.use_specified_kvno: 'PA-USE-SPECIFIED-KVNO or PA-SVR-REFERRAL-INFO', KerberosPADataType.sam_redirect: 'PA-SAM-REDIRECT', KerberosPADataType.get_from_typed_data: 'PA-GET-FROM-TYPED-DATA', KerberosPADataType.td_padata: 'TD-PADATA', KerberosPADataType.sam_etype_info: 'PA-SAM-ETYPE-INFO', KerberosPADataType.alt_princ: 'PA-ALT-PRINC', KerberosPADataType.server_referral: 'PA-SERVER-REFERRAL', KerberosPADataType.sam_challenge2: 'PA-SAM-CHALLENGE2', KerberosPADataType.sam_response2: 'PA-SAM-RESPONSE2', KerberosPADataType.extra_tgt: 'PA-EXTRA-TGT', KerberosPADataType.td_pkinit_cms_certificates: 'TD-PKINIT-CMS-CERTIFICATES', KerberosPADataType.td_krb_principal: 'TD-KRB-PRINCIPAL', KerberosPADataType.td_krb_realm: 'TD-KRB-REALM', KerberosPADataType.td_trusted_certifiers: 'TD-TRUSTED-CERTIFIERS', KerberosPADataType.td_certificate_index: 'TD-CERTIFICATE-INDEX', KerberosPADataType.td_app_defined_error: 'TD-APP-DEFINED-ERROR', KerberosPADataType.td_req_nonce: 'TD-REQ-NONCE', KerberosPADataType.td_req_seq: 'TD-REQ-SEQ', KerberosPADataType.td_dh_parameters: 'TD_DH_PARAMETERS', KerberosPADataType.td_cms_digest_algorithms: 'TD-CMS-DIGEST-ALGORITHMS', KerberosPADataType.td_cert_digest_algorithms: 'TD-CERT-DIGEST-ALGORITHMS', KerberosPADataType.pac_request: 'PA-PAC-REQUEST', KerberosPADataType.for_user: 'PA-FOR_USER', KerberosPADataType.for_x509_user: 'PA-FOR-X509-USER', KerberosPADataType.for_check_dups: 'PA-FOR-CHECK_DUPS', KerberosPADataType.as_checksum: 'PA-AS-CHECKSUM', KerberosPADataType.fx_cookie: 'PA-FX-COOKIE', KerberosPADataType.authentication_set: 'PA-AUTHENTICATION-SET', KerberosPADataType.auth_set_selected: 'PA-AUTH-SET-SELECTED', KerberosPADataType.fx_fast: 'PA-FX-FAST', KerberosPADataType.fx_error: 'PA-FX-ERROR', KerberosPADataType.encrypted_challenge: 'PA-ENCRYPTED-CHALLENGE', KerberosPADataType.otp_challenge: 'PA-OTP-CHALLENGE', KerberosPADataType.otp_request: 'PA-OTP-REQUEST', KerberosPADataType.otp_confirm: 'PA-OTP-CONFIRM', KerberosPADataType.otp_pin_change: 'PA-OTP-PIN-CHANGE', KerberosPADataType.epak_as_req: 'PA-EPAK-AS-REQ', KerberosPADataType.epak_as_rep: 'PA-EPAK-AS-REP', KerberosPADataType.pkinit_kx: 'PA_PKINIT_KX', KerberosPADataType.pku2u_name: 'PA_PKU2U_NAME', KerberosPADataType.enc_pa_rep: 'PA-REQ-ENC-PA-REP', KerberosPADataType.as_freshness: 'PA_AS_FRESHNESS', KerberosPADataType.spake: 'PA-SPAKE', KerberosPADataType.kerb_key_list_req: 'KERB-KEY-LIST-REQ', KerberosPADataType.kerb_key_list_rep: 'KERB-KEY-LIST-REP', KerberosPADataType.supported_etypes: 'PA-SUPPORTED-ETYPES', KerberosPADataType.extended_error: 'PA-EXTENDED_ERROR', KerberosPADataType.pac_options: 'PA-PAC-OPTIONS'}