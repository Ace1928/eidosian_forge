import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
class GLError(Error):
    """OpenGL core error implementation class
    
    Primary purpose of this error class is to allow for 
    annotating an error with more details about the calling 
    environment so that it's easier to debug errors in the
    wrapping process.
    
    Attributes:
    
        err -- the OpenGL error code for the error 
        result -- the OpenGL result code for the operation
        baseOperation -- the "function" being called
        pyArgs -- the translated set of Python arguments
        cArgs -- the Python objects matching 1:1 the C arguments
        cArguments -- ctypes-level arguments to the operation,
            often raw integers for pointers and the like
        description -- OpenGL description of the error (textual)
    """

    def __init__(self, err=None, result=None, cArguments=None, baseOperation=None, pyArgs=None, cArgs=None, description=None):
        """Initialise the GLError, storing metadata for later display"""
        self.err, self.result, self.cArguments, self.baseOperation, self.pyArgs, self.cArgs, self.description = (err, result, cArguments, baseOperation, pyArgs, cArgs, description)
    DISPLAY_ORDER = ('err', 'description', 'baseOperation', 'pyArgs', 'cArgs', 'cArguments', 'result')

    def __str__(self):
        """Create a fully formatted representation of the error"""
        args = []
        for property in self.DISPLAY_ORDER:
            value = getattr(self, property, None)
            if value is not None or property == 'description':
                formatFunction = 'format_%s' % property
                if hasattr(self, formatFunction):
                    args.append(getattr(self, formatFunction)(property, value))
                else:
                    args.append('%s = %s' % (property, self.shortRepr(value)))
        return '%s(\n\t%s\n)' % (self.__class__.__name__, ',\n\t'.join([x for x in args if x]))

    def __repr__(self):
        """Produce a much shorter version of the error as a string"""
        return '%s( %s )' % (self.__class__.__name__, ', '.join([x for x in ['err=%s' % self.err, self.format_description('description', self.description) or '', self.format_baseOperation('baseOperation', self.baseOperation) or ''] if x]))

    def format_description(self, property, value):
        """Format description using GLU's gluErrorString"""
        if value is None and self.err is not None:
            try:
                from OpenGL.GLU import gluErrorString
                self.description = value = gluErrorString(self.err)
            except Exception as err:
                return None
        if value is None:
            return None
        return '%s = %s' % (property, self.shortRepr(value))

    def shortRepr(self, value, firstLevel=True):
        """Retrieve short representation of the given value"""
        if isinstance(value, (list, tuple)) and value and (len(repr(value)) >= 40):
            if isinstance(value, list):
                template = '[\n\t\t%s\n\t]'
            else:
                template = '(\n\t\t%s,\n\t)'
            return template % ',\n\t\t'.join([self.shortRepr(x, False) for x in value])
        r = repr(value)
        if len(r) < 120:
            return r
        else:
            return r[:117] + '...'

    def format_baseOperation(self, property, value):
        """Format a baseOperation reference for display"""
        if hasattr(value, '__name__'):
            return '%s = %s' % (property, value.__name__)
        else:
            return '%s = %r' % (property, value)