import sys
from pyu2f.convenience import baseauthenticator
from pyu2f.convenience import customauthenticator
from pyu2f.convenience import localauthenticator
class CompositeAuthenticator(baseauthenticator.BaseAuthenticator):
    """Composes multiple authenticators into a single authenticator.

  Priority is based on the order of the list initialized with the instance.
  """

    def __init__(self, authenticators):
        self.authenticators = authenticators

    def Authenticate(self, app_id, challenge_data, print_callback=sys.stderr.write):
        """See base class."""
        for authenticator in self.authenticators:
            if authenticator.IsAvailable():
                result = authenticator.Authenticate(app_id, challenge_data, print_callback)
                return result
        raise ValueError('No valid authenticators found')

    def IsAvailable(self):
        """See base class."""
        return True