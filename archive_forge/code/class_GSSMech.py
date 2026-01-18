import abc
import dataclasses
import enum
import typing
import warnings
from spnego._credential import Credential
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import FeatureMissingError, NegotiateOptions, SpnegoError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
class GSSMech(str, enum.Enum):
    ntlm = '1.3.6.1.4.1.311.2.2.10'
    spnego = '1.3.6.1.5.5.2'
    kerberos = '1.2.840.113554.1.2.2'
    _ms_kerberos = '1.2.840.48018.1.2.2'
    _kerberos_draft = '1.3.5.1.5.2'
    _iakerb = '1.3.6.1.5.2'
    kerberos_u2u = '1.2.840.113554.1.2.2.3'
    negoex = '1.3.6.1.4.1.311.2.2.30'

    @classmethod
    def native_labels(cls) -> typing.Dict[str, str]:
        return {GSSMech.ntlm: 'NTLM', GSSMech.ntlm.value: 'NTLM', GSSMech.spnego: 'SPNEGO', GSSMech.spnego.value: 'SPNEGO', GSSMech.kerberos: 'Kerberos', GSSMech.kerberos.value: 'Kerberos', GSSMech._ms_kerberos: 'MS Kerberos', GSSMech._ms_kerberos.value: 'MS Kerberos', GSSMech._kerberos_draft: 'Kerberos (draft)', GSSMech._kerberos_draft.value: 'Kerberos (draft)', GSSMech._iakerb: 'IAKerberos', GSSMech._iakerb.value: 'IAKerberos', GSSMech.kerberos_u2u: 'Kerberos User to User', GSSMech.kerberos_u2u.value: 'Kerberos User to User', GSSMech.negoex: 'NEGOEX', GSSMech.negoex.value: 'NEGOEX'}

    @property
    def common_name(self) -> str:
        if self.is_kerberos_oid:
            return 'kerberos'
        return self.name

    @property
    def is_kerberos_oid(self) -> bool:
        """Determines if the mech is a Kerberos mech.

        Kerberos has been known under serveral OIDs in the past. This tells the caller whether the OID is one of those
        "known" OIDs.

        Returns:
            bool: Whether the mech is a Kerberos mech (True) or not (False).
        """
        return self in [GSSMech.kerberos, GSSMech._ms_kerberos, GSSMech._kerberos_draft, GSSMech._iakerb]

    @staticmethod
    def from_oid(oid: str) -> 'GSSMech':
        """Converts an OID string to a GSSMech value.

        Converts an OID string to a GSSMech value if it is known.

        Args:
            oid: The OID as a string to convert from.

        Raises:
            ValueError: if the OID is not a known GSSMech.
        """
        for mech in GSSMech:
            if mech.value == oid:
                return mech
        else:
            raise ValueError("'%s' is not a valid GSSMech OID" % oid)