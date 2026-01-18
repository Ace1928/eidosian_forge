class FixtureValidationError(Exception):
    """The token you created is not legitimate.

    The data contained in the token that was generated is not valid and would
    not have been returned from a keystone server. You should not do testing
    with this token.
    """