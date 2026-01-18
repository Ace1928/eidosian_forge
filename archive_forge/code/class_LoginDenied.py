class LoginDenied(LoginFailed):
    """
    The realm rejected this login for some reason.

    Examples of reasons this might be raised include an avatar logging in
    too frequently, a quota having been fully used, or the overall server
    load being too high.
    """