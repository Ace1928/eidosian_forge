import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
class _SSH_GSSAPI_NEW(_SSH_GSSAuth):
    """
    Implementation of the GSS-API MIT Kerberos Authentication for SSH2,
    using the newer, currently maintained gssapi package.

    :see: `.GSSAuth`
    """

    def __init__(self, auth_method, gss_deleg_creds):
        """
        :param str auth_method: The name of the SSH authentication mechanism
                                (gssapi-with-mic or gss-keyex)
        :param bool gss_deleg_creds: Delegate client credentials or not
        """
        _SSH_GSSAuth.__init__(self, auth_method, gss_deleg_creds)
        if self._gss_deleg_creds:
            self._gss_flags = (gssapi.RequirementFlag.protection_ready, gssapi.RequirementFlag.integrity, gssapi.RequirementFlag.mutual_authentication, gssapi.RequirementFlag.delegate_to_peer)
        else:
            self._gss_flags = (gssapi.RequirementFlag.protection_ready, gssapi.RequirementFlag.integrity, gssapi.RequirementFlag.mutual_authentication)

    def ssh_init_sec_context(self, target, desired_mech=None, username=None, recv_token=None):
        """
        Initialize a GSS-API context.

        :param str username: The name of the user who attempts to login
        :param str target: The hostname of the target to connect to
        :param str desired_mech: The negotiated GSS-API mechanism
                                 ("pseudo negotiated" mechanism, because we
                                 support just the krb5 mechanism :-))
        :param str recv_token: The GSS-API token received from the Server
        :raises: `.SSHException` -- Is raised if the desired mechanism of the
                 client is not supported
        :raises: ``gssapi.exceptions.GSSError`` if there is an error signaled
                                                by the GSS-API implementation
        :return: A ``String`` if the GSS-API has returned a token or ``None``
                 if no token was returned
        """
        from pyasn1.codec.der import decoder
        self._username = username
        self._gss_host = target
        targ_name = gssapi.Name('host@' + self._gss_host, name_type=gssapi.NameType.hostbased_service)
        if desired_mech is not None:
            mech, __ = decoder.decode(desired_mech)
            if mech.__str__() != self._krb5_mech:
                raise SSHException('Unsupported mechanism OID.')
        krb5_mech = gssapi.MechType.kerberos
        token = None
        if recv_token is None:
            self._gss_ctxt = gssapi.SecurityContext(name=targ_name, flags=self._gss_flags, mech=krb5_mech, usage='initiate')
            token = self._gss_ctxt.step(token)
        else:
            token = self._gss_ctxt.step(recv_token)
        self._gss_ctxt_status = self._gss_ctxt.complete
        return token

    def ssh_get_mic(self, session_id, gss_kex=False):
        """
        Create the MIC token for a SSH2 message.

        :param str session_id: The SSH session ID
        :param bool gss_kex: Generate the MIC for GSS-API Key Exchange or not
        :return: gssapi-with-mic:
                 Returns the MIC token from GSS-API for the message we created
                 with ``_ssh_build_mic``.
                 gssapi-keyex:
                 Returns the MIC token from GSS-API with the SSH session ID as
                 message.
        :rtype: str
        """
        self._session_id = session_id
        if not gss_kex:
            mic_field = self._ssh_build_mic(self._session_id, self._username, self._service, self._auth_method)
            mic_token = self._gss_ctxt.get_signature(mic_field)
        else:
            mic_token = self._gss_srv_ctxt.get_signature(self._session_id)
        return mic_token

    def ssh_accept_sec_context(self, hostname, recv_token, username=None):
        """
        Accept a GSS-API context (server mode).

        :param str hostname: The servers hostname
        :param str username: The name of the user who attempts to login
        :param str recv_token: The GSS-API Token received from the server,
                               if it's not the initial call.
        :return: A ``String`` if the GSS-API has returned a token or ``None``
                if no token was returned
        """
        self._gss_host = hostname
        self._username = username
        if self._gss_srv_ctxt is None:
            self._gss_srv_ctxt = gssapi.SecurityContext(usage='accept')
        token = self._gss_srv_ctxt.step(recv_token)
        self._gss_srv_ctxt_status = self._gss_srv_ctxt.complete
        return token

    def ssh_check_mic(self, mic_token, session_id, username=None):
        """
        Verify the MIC token for a SSH2 message.

        :param str mic_token: The MIC token received from the client
        :param str session_id: The SSH session ID
        :param str username: The name of the user who attempts to login
        :return: None if the MIC check was successful
        :raises: ``gssapi.exceptions.GSSError`` -- if the MIC check failed
        """
        self._session_id = session_id
        self._username = username
        if self._username is not None:
            mic_field = self._ssh_build_mic(self._session_id, self._username, self._service, self._auth_method)
            self._gss_srv_ctxt.verify_signature(mic_field, mic_token)
        else:
            self._gss_ctxt.verify_signature(self._session_id, mic_token)

    @property
    def credentials_delegated(self):
        """
        Checks if credentials are delegated (server mode).

        :return: ``True`` if credentials are delegated, otherwise ``False``
        :rtype: bool
        """
        if self._gss_srv_ctxt.delegated_creds is not None:
            return True
        return False

    def save_client_creds(self, client_token):
        """
        Save the Client token in a file. This is used by the SSH server
        to store the client credentials if credentials are delegated
        (server mode).

        :param str client_token: The GSS-API token received form the client
        :raises: ``NotImplementedError`` -- Credential delegation is currently
                 not supported in server mode
        """
        raise NotImplementedError