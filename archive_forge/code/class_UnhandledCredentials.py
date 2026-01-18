class UnhandledCredentials(LoginFailed):
    """A type of credentials were passed in with no knowledge of how to check
    them.  This is a server configuration error - it means that a protocol was
    connected to a Portal without a CredentialChecker that can check all of its
    potential authentication strategies.
    """