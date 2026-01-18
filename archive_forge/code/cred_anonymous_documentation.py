from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import AllowAnonymousAccess
from twisted.cred.credentials import IAnonymous
from twisted.cred.strcred import ICheckerFactory

    Generates checkers that will authenticate an anonymous request.
    