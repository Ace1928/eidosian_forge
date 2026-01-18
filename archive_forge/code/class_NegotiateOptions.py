import enum
import typing
class NegotiateOptions(enum.IntFlag):
    """Flags that the caller can use to control the negotiation behaviour.

    A list of features as bit flags that the caller can specify when creating the security context. These flags can
    be used on both Windows or Linux but are a no-op on Windows as it should always have the same features available.
    On Linux the features it can implement depend on a wide range of factors like the system libraries/headers that
    are installed, what GSSAPI implementation is present, and what Python libraries are available.

    This is a pretty advanced feature and is mostly a way to control the kerberos to ntlm fallback behaviour on Linux.
    The `use_*` options when combined with `protocol='credssp'` control the underlying auth proxy provider that is used
    in the CredSSP authentication process and not the parent context proxy the user interacts with.

    These are the currently implemented feature flags:

    use_sspi:
        Ensures the context proxy used is :class:`spnego._sspi.SSPIProxy`.

    use_gssapi:
        Ensures the context proxy used is :class:`spnego._gss.GSSAPIProxy`.

    use_negotiate:
        Ensures the context proxy used is :class:`spnego._negotiate.NegotiateProxy`.

    use_ntlm:
        Ensures the context proxy used is :class:`spnego._ntlm.NTLMProxy`.

    negotiate_kerberos:
        Will make sure that Kerberos is at least available to try for authentication when using the `negotiate`
        protocol. If Kerberos cannot be used due to the Python gssapi library not being installed then it will raise a
        :class:`spnego.exceptions.FeatureMissingError`. If Kerberos was available but it cannot get a credential or
        create a context then it will just fallback to NTLM auth. If you wish to only use Kerberos with no NTLM
        fallback, set `protocol='kerberos'` when creating the security context.

    session_key:
        Ensure that the authenticated context will be able to return the session key that was negotiated between the
        client and the server. Older versions of `gss-ntlmssp`_ do not expose the functions required to retrieve this
        info so when this feature flag is set then the NTLM fallback process will use a builtin NTLM process and not
        `gss-ntlmssp`_ if the latter is too old to retrieve the session key. Cannot be used in combination with
        `protocol='credssp'` as CredSSP does not provide a session key.

    wrapping_iov:
        The GSSAPI IOV methods are extensions to the Kerberos spec and not implemented or exposed on all platforms,
        macOS is a popular example. If the caller requires the wrap_iov and unwrap_iov methods this will ensure it
        fails fast before the auth has been set up. Unfortunately there is no fallback for this as if the headers
        aren't present for GSSAPI then we can't do anything to fix that. This won't fail if `negotiate` was used and
        NTLM was the chosen protocol as that happens post negotiation.

    wrapping_winrm:
        To created a wrapped WinRM message the IOV extensions are required when using Kerberos auth. Setting this flag
        will skip Kerberos when `protocol='negotiate'` if the IOV headers aren't present and just fallback to NTLM.

    .. _gss-ntlmssp:
        https://github.com/gssapi/gss-ntlmssp
    """
    none = 0
    use_sspi = 1
    use_gssapi = 2
    use_negotiate = 4
    use_ntlm = 8
    negotiate_kerberos = 16
    session_key = 32
    wrapping_iov = 64
    wrapping_winrm = 128