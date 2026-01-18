class UnauthorizedLogin(LoginFailed, Unauthorized):
    """The user was not authorized to log in."""