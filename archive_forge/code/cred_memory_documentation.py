from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import IUsernameHashedPassword, IUsernamePassword
from twisted.cred.strcred import ICheckerFactory

        This checker factory expects to get a list of
        username:password pairs, with each pair also separated by a
        colon. For example, the string 'alice:f:bob:g' would generate
        two users, one named 'alice' and one named 'bob'.
        