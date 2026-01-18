from typing import Optional
class InsecureAuthenticationDisallowed(POP3ClientError):
    """
    An error indicating secure authentication was required but no mechanism
    could be found.
    """