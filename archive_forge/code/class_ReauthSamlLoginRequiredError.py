class ReauthSamlLoginRequiredError(ReauthError):
    """An exception for when web login is required to complete reauth.

    This applies to SAML users who are required to login through their IDP to
    complete reauth.
    """

    def __init__(self):
        super(ReauthSamlLoginRequiredError, self).__init__('SAML login is required for the current account to complete reauthentication.')