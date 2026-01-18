from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.strcred import ICheckerFactory
from twisted.internet import defer

        This checker factory ignores the argument string. Everything
        needed to generate a user database is pulled out of the local
        UNIX environment.
        