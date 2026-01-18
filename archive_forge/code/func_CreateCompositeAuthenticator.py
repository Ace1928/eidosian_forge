import sys
from pyu2f.convenience import baseauthenticator
from pyu2f.convenience import customauthenticator
from pyu2f.convenience import localauthenticator
def CreateCompositeAuthenticator(origin):
    authenticators = [customauthenticator.CustomAuthenticator(origin), localauthenticator.LocalAuthenticator(origin)]
    return CompositeAuthenticator(authenticators)