from zope.interface import implementer
from twisted import plugin
from twisted.cred.strcred import ICheckerFactory

            This checker factory ignores the argument string. Everything
            needed to authenticate users is pulled out of the public keys
            listed in user .ssh/ directories.
            