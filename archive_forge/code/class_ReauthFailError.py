class ReauthFailError(ReauthError):
    """An exception for when reauth failed."""

    def __init__(self, message=None):
        super(ReauthFailError, self).__init__('Reauthentication challenge failed. {0}'.format(message))