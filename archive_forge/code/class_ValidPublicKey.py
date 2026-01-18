from twisted.cred.error import UnauthorizedLogin
class ValidPublicKey(UnauthorizedLogin):
    """
    Raised by public key checkers when they receive public key credentials
    that don't contain a signature at all, but are valid in every other way.
    (e.g. the public key matches one in the user's authorized_keys file).

    Protocol code (eg
    L{SSHUserAuthServer<twisted.conch.ssh.userauth.SSHUserAuthServer>}) which
    attempts to log in using
    L{ISSHPrivateKey<twisted.cred.credentials.ISSHPrivateKey>} credentials
    should be prepared to handle a failure of this type by telling the user to
    re-authenticate using the same key and to include a signature with the new
    attempt.

    See U{http://www.ietf.org/rfc/rfc4252.txt} section 7 for more details.
    """