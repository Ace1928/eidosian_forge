import enum
import typing
class FeatureMissingError(Exception):

    @property
    def feature_id(self) -> NegotiateOptions:
        return self.args[0]

    @property
    def message(self) -> str:
        msg = {NegotiateOptions.negotiate_kerberos: 'The Python gssapi library is not installed so Kerberos cannot be negotiated.', NegotiateOptions.wrapping_iov: 'The system is missing the GSSAPI IOV extension headers or CredSSP is being requested, cannot utilize wrap_iov and unwrap_iov', NegotiateOptions.wrapping_winrm: 'The system is missing the GSSAPI IOV extension headers required for WinRM encryption with Kerberos.', NegotiateOptions.session_key: 'The protocol selected does not support getting the session key.'}.get(self.feature_id, 'Unknown option flag: %d' % self.feature_id)
        return msg

    def __str__(self) -> str:
        return self.message