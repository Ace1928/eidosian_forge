from zope.interface import implementer
from twisted import plugin
from twisted.cred.strcred import ICheckerFactory
@implementer(ICheckerFactory, plugin.IPlugin)
class SSHKeyCheckerFactory:
    """
        Generates checkers that will authenticate a SSH public key
        """
    authType = 'sshkey'
    authHelp = sshKeyCheckerFactoryHelp
    argStringFormat = 'No argstring required.'
    credentialInterfaces = SSHPublicKeyChecker.credentialInterfaces

    def generateChecker(self, argstring=''):
        """
            This checker factory ignores the argument string. Everything
            needed to authenticate users is pulled out of the public keys
            listed in user .ssh/ directories.
            """
        return SSHPublicKeyChecker(UNIXAuthorizedKeysFiles())