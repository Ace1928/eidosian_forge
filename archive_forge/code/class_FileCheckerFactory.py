import sys
from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import FilePasswordDB
from twisted.cred.credentials import IUsernameHashedPassword, IUsernamePassword
from twisted.cred.strcred import ICheckerFactory
@implementer(ICheckerFactory, plugin.IPlugin)
class FileCheckerFactory:
    """
    A factory for instances of L{FilePasswordDB}.
    """
    authType = 'file'
    authHelp = fileCheckerFactoryHelp
    argStringFormat = 'Location of a FilePasswordDB-formatted file.'
    credentialInterfaces = (IUsernamePassword, IUsernameHashedPassword)
    errorOutput = sys.stderr

    def generateChecker(self, argstring):
        """
        This checker factory expects to get the location of a file.
        The file should conform to the format required by
        L{FilePasswordDB} (using defaults for all
        initialization parameters).
        """
        from twisted.python.filepath import FilePath
        if not argstring.strip():
            raise ValueError('%r requires a filename' % self.authType)
        elif not FilePath(argstring).isfile():
            self.errorOutput.write(f'{invalidFileWarning}: {argstring}\n')
        return FilePasswordDB(argstring)