import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
class _ErrorChecker(object):
    """Per-API error-checking object
            
            Attributes:
                _registeredChecker -- the checking function enabled when 
                    not doing onBegin/onEnd processing
                _currentChecker -- currently active checking function
            """
    _getErrors = None

    def __init__(self, platform, baseOperation=None, noErrorResult=0, errorClass=GLError):
        """Initialize from a platform module/reference"""
        self._isValid = platform.CurrentContextIsValid
        self._getErrors = baseOperation
        self._noErrorResult = noErrorResult
        self._errorClass = errorClass
        if self._getErrors:
            if _configflags.CONTEXT_CHECKING:
                self._registeredChecker = self.safeGetError
            else:
                self._registeredChecker = self._getErrors
        else:
            self._registeredChecker = self.nullGetError
        self._currentChecker = self._registeredChecker

    def __bool__(self):
        """We are "true" if we actually do anything"""
        if self._registeredChecker is self.nullGetError:
            return False
        return True

    def safeGetError(self):
        """Check for error, testing for context before operation"""
        if self._isValid():
            return self._getErrors()
        return None

    def nullGetError(self):
        """Used as error-checker when no error checking should be done"""
        return self._noErrorResult

    def glCheckError(self, result, baseOperation=None, cArguments=None, *args):
        """Base GL Error checker compatible with new ctypes errcheck protocol
                
                This function will raise a GLError with just the calling information
                available at the C-calling level, i.e. the error code, cArguments,
                baseOperation and result.  Higher-level code is responsible for any 
                extra annotations.
                
                Note:
                    glCheckError relies on glBegin/glEnd interactions to 
                    prevent glGetError being called during a glBegin/glEnd 
                    sequence.  If you are calling glBegin/glEnd in C you 
                    should call onBegin and onEnd appropriately.
                """
        err = self._currentChecker()
        if err != self._noErrorResult:
            raise self._errorClass(err, result, cArguments=cArguments, baseOperation=baseOperation)
        return result

    def onBegin(self):
        """Called by glBegin to record the fact that glGetError won't work"""
        self._currentChecker = self.nullGetError

    def onEnd(self):
        """Called by glEnd to record the fact that glGetError will work"""
        self._currentChecker = self._registeredChecker