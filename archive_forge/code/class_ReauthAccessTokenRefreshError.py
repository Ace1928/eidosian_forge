class ReauthAccessTokenRefreshError(ReauthError):
    """An exception for when we can't get an access token for reauth."""

    def __init__(self, message=None, status=None):
        super(ReauthAccessTokenRefreshError, self).__init__('Failed to get an access token for reauthentication. {0}'.format(message))
        self.status = status