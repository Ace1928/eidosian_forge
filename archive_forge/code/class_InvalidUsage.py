class InvalidUsage(RuntimeError):
    """This function call is invalid in the way you are using this client.

    Due to the transition to using keystoneauth some function calls are no
    longer available. You should make a similar call to the session object
    instead.
    """
    pass