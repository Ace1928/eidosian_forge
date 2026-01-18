from twisted.cred.error import UnauthorizedLogin
class IgnoreAuthentication(Exception):
    """
    This is thrown to let the UserAuthServer know it doesn't need to handle the
    authentication anymore.
    """