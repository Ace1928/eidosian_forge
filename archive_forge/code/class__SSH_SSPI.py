import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
class _SSH_SSPI(_SSH_GSSAuth):
    """
    Implementation of the Microsoft SSPI Kerberos Authentication for SSH2.

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
            self._gss_flags = sspicon.ISC_REQ_INTEGRITY | sspicon.ISC_REQ_MUTUAL_AUTH | sspicon.ISC_REQ_DELEGATE
        else:
            self._gss_flags = sspicon.ISC_REQ_INTEGRITY | sspicon.ISC_REQ_MUTUAL_AUTH

    def ssh_init_sec_context(self, target, desired_mech=None, username=None, recv_token=None):
        """
        Initialize a SSPI context.

        :param str username: The name of the user who attempts to login
        :param str target: The FQDN of the target to connect to
        :param str desired_mech: The negotiated SSPI mechanism
                                 ("pseudo negotiated" mechanism, because we
                                 support just the krb5 mechanism :-))
        :param recv_token: The SSPI token received from the Server
        :raises:
            `.SSHException` -- Is raised if the desired mechanism of the client
            is not supported
        :return: A ``String`` if the SSPI has returned a token or ``None`` if
                 no token was returned
        """
        from pyasn1.codec.der import decoder
        self._username = username
        self._gss_host = target
        error = 0
        targ_name = 'host/' + self._gss_host
        if desired_mech is not None:
            mech, __ = decoder.decode(desired_mech)
            if mech.__str__() != self._krb5_mech:
                raise SSHException('Unsupported mechanism OID.')
        try:
            if recv_token is None:
                self._gss_ctxt = sspi.ClientAuth('Kerberos', scflags=self._gss_flags, targetspn=targ_name)
            error, token = self._gss_ctxt.authorize(recv_token)
            token = token[0].Buffer
        except pywintypes.error as e:
            e.strerror += ', Target: {}'.format(self._gss_host)
            raise
        if error == 0:
            '\n            if the status is GSS_COMPLETE (error = 0) the context is fully\n            established an we can set _gss_ctxt_status to True.\n            '
            self._gss_ctxt_status = True
            token = None
            '\n            You won\'t get another token if the context is fully established,\n            so i set token to None instead of ""\n            '
        return token

    def ssh_get_mic(self, session_id, gss_kex=False):
        """
        Create the MIC token for a SSH2 message.

        :param str session_id: The SSH session ID
        :param bool gss_kex: Generate the MIC for Key Exchange with SSPI or not
        :return: gssapi-with-mic:
                 Returns the MIC token from SSPI for the message we created
                 with ``_ssh_build_mic``.
                 gssapi-keyex:
                 Returns the MIC token from SSPI with the SSH session ID as
                 message.
        """
        self._session_id = session_id
        if not gss_kex:
            mic_field = self._ssh_build_mic(self._session_id, self._username, self._service, self._auth_method)
            mic_token = self._gss_ctxt.sign(mic_field)
        else:
            mic_token = self._gss_srv_ctxt.sign(self._session_id)
        return mic_token

    def ssh_accept_sec_context(self, hostname, username, recv_token):
        """
        Accept a SSPI context (server mode).

        :param str hostname: The servers FQDN
        :param str username: The name of the user who attempts to login
        :param str recv_token: The SSPI Token received from the server,
                               if it's not the initial call.
        :return: A ``String`` if the SSPI has returned a token or ``None`` if
                 no token was returned
        """
        self._gss_host = hostname
        self._username = username
        targ_name = 'host/' + self._gss_host
        self._gss_srv_ctxt = sspi.ServerAuth('Kerberos', spn=targ_name)
        error, token = self._gss_srv_ctxt.authorize(recv_token)
        token = token[0].Buffer
        if error == 0:
            self._gss_srv_ctxt_status = True
            token = None
        return token

    def ssh_check_mic(self, mic_token, session_id, username=None):
        """
        Verify the MIC token for a SSH2 message.

        :param str mic_token: The MIC token received from the client
        :param str session_id: The SSH session ID
        :param str username: The name of the user who attempts to login
        :return: None if the MIC check was successful
        :raises: ``sspi.error`` -- if the MIC check failed
        """
        self._session_id = session_id
        self._username = username
        if username is not None:
            mic_field = self._ssh_build_mic(self._session_id, self._username, self._service, self._auth_method)
            self._gss_srv_ctxt.verify(mic_field, mic_token)
        else:
            self._gss_ctxt.verify(self._session_id, mic_token)

    @property
    def credentials_delegated(self):
        """
        Checks if credentials are delegated (server mode).

        :return: ``True`` if credentials are delegated, otherwise ``False``
        """
        return self._gss_flags & sspicon.ISC_REQ_DELEGATE and (self._gss_srv_ctxt_status or self._gss_flags)

    def save_client_creds(self, client_token):
        """
        Save the Client token in a file. This is used by the SSH server
        to store the client credentails if credentials are delegated
        (server mode).

        :param str client_token: The SSPI token received form the client
        :raises:
            ``NotImplementedError`` -- Credential delegation is currently not
            supported in server mode
        """
        raise NotImplementedError