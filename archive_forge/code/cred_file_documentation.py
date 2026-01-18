import sys
from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import FilePasswordDB
from twisted.cred.credentials import IUsernameHashedPassword, IUsernamePassword
from twisted.cred.strcred import ICheckerFactory

        This checker factory expects to get the location of a file.
        The file should conform to the format required by
        L{FilePasswordDB} (using defaults for all
        initialization parameters).
        