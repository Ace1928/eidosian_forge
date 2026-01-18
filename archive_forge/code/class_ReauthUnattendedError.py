class ReauthUnattendedError(ReauthError):
    """An exception for when reauth cannot be answered."""

    def __init__(self):
        super(ReauthUnattendedError, self).__init__('Reauthentication challenge could not be answered because you are not in an interactive session.')