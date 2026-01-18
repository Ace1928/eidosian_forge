from twisted.application import strports
from twisted.conch import checkers as conch_checkers, unix
from twisted.conch.openssh_compat import factory
from twisted.cred import portal, strcred
from twisted.python import usage
def addChecker(self, checker):
    """
        Add the checker specified.  If any checkers are added, the default
        checkers are automatically cleared and the only checkers will be the
        specified one(s).
        """
    if self._usingDefaultAuth:
        self['credCheckers'] = []
        self['credInterfaces'] = {}
        self._usingDefaultAuth = False
    super().addChecker(checker)