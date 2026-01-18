import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
class BasePlatform(object):
    """Base class for per-platform implementations
    
    Attributes of note:
    
        EXPORTED_NAMES -- set of names exported via the platform 
            module's namespace...
    
        GL, GLU, GLUT, GLE, GLES1, GLES2, GLES3 -- ctypes libraries
    
        DEFAULT_FUNCTION_TYPE -- used as the default function 
            type for functions unless overridden on a per-DLL
            basis with a "FunctionType" member
        
        GLUT_GUARD_CALLBACKS -- if True, the GLUT wrappers 
            will provide guarding wrappers to prevent GLUT 
            errors with uninitialised GLUT.
        
        EXTENSIONS_USE_BASE_FUNCTIONS -- if True, uses regular
            dll attribute-based lookup to retrieve extension 
            function pointers.
    """
    EXPORTED_NAMES = ['GetCurrentContext', 'CurrentContextIsValid', 'createBaseFunction', 'createExtensionFunction', 'copyBaseFunction', 'getGLUTFontPointer', 'nullFunction', 'GLUT_GUARD_CALLBACKS']
    DEFAULT_FUNCTION_TYPE = None
    GLUT_GUARD_CALLBACKS = False
    EXTENSIONS_USE_BASE_FUNCTIONS = False

    def install(self, namespace):
        """Install this platform instance into the platform module"""
        for name in self.EXPORTED_NAMES:
            namespace[name] = getattr(self, name, None)
        namespace['PLATFORM'] = self
        return self

    def functionTypeFor(self, dll):
        """Given a DLL, determine appropriate function type..."""
        if hasattr(dll, 'FunctionType'):
            return dll.FunctionType
        else:
            return self.DEFAULT_FUNCTION_TYPE

    def errorChecking(self, func, dll, error_checker=None):
        """Add error checking to the function if appropriate"""
        from OpenGL import error
        if error_checker and _configflags.ERROR_CHECKING:
            func.errcheck = error_checker.glCheckError
        return func

    def wrapContextCheck(self, func, dll):
        """Wrap function with context-checking if appropriate"""
        if _configflags.CONTEXT_CHECKING and dll is self.GL and (func.__name__ not in ('glGetString', 'glGetStringi', 'glGetIntegerv')) and (not func.__name__.startswith('glX')):
            return _CheckContext(func, self.CurrentContextIsValid)
        return func

    def wrapLogging(self, func):
        """Wrap function with logging operations if appropriate"""
        return logs.logOnFail(func, logs.getLog('OpenGL.errors'))

    def finalArgType(self, typ):
        """Retrieve a final type for arg-type"""
        if typ == ctypes.POINTER(None) and (not getattr(typ, 'final', False)):
            from OpenGL.arrays import ArrayDatatype
            return ArrayDatatype
        else:
            return typ

    def constructFunction(self, functionName, dll, resultType=ctypes.c_int, argTypes=(), doc=None, argNames=(), extension=None, deprecated=False, module=None, force_extension=False, error_checker=None):
        """Core operation to create a new base ctypes function
        
        raises AttributeError if can't find the procedure...
        """
        is_core = not extension or extension.split('_')[1] == 'VERSION'
        if not is_core and (not self.checkExtension(extension)):
            raise AttributeError('Extension not available')
        argTypes = [self.finalArgType(t) for t in argTypes]
        if force_extension or (not is_core and (not self.EXTENSIONS_USE_BASE_FUNCTIONS)):
            pointer = self.getExtensionProcedure(as_8_bit(functionName))
            if pointer:
                func = self.functionTypeFor(dll)(resultType, *argTypes)(pointer)
            else:
                raise AttributeError('Extension %r available, but no pointer for function %r' % (extension, functionName))
        else:
            func = ctypesloader.buildFunction(self.functionTypeFor(dll)(resultType, *argTypes), functionName, dll)
        func.__doc__ = doc
        func.argNames = list(argNames or ())
        func.__name__ = functionName
        func.DLL = dll
        func.extension = extension
        func.deprecated = deprecated
        func = self.wrapLogging(self.wrapContextCheck(self.errorChecking(func, dll, error_checker=error_checker), dll))
        if MODULE_ANNOTATIONS:
            if not module:
                module = _find_module()
            if module:
                func.__module__ = module
        return func

    def createBaseFunction(self, functionName, dll, resultType=ctypes.c_int, argTypes=(), doc=None, argNames=(), extension=None, deprecated=False, module=None, error_checker=None):
        """Create a base function for given name
        
        Normally you can just use the dll.name hook to get the object,
        but we want to be able to create different bindings for the 
        same function, so we do the work manually here to produce a
        base function from a DLL.
        """
        from OpenGL import wrapper
        result = None
        try:
            if _configflags.FORWARD_COMPATIBLE_ONLY and dll is self.GL and deprecated:
                result = self.nullFunction(functionName, dll=dll, resultType=resultType, argTypes=argTypes, doc=doc, argNames=argNames, extension=extension, deprecated=deprecated, error_checker=error_checker)
            else:
                result = self.constructFunction(functionName, dll, resultType=resultType, argTypes=argTypes, doc=doc, argNames=argNames, extension=extension, error_checker=error_checker)
        except AttributeError as err:
            result = self.nullFunction(functionName, dll=dll, resultType=resultType, argTypes=argTypes, doc=doc, argNames=argNames, extension=extension, error_checker=error_checker)
        if MODULE_ANNOTATIONS:
            if not module:
                module = _find_module()
            if module:
                result.__module__ = module
        return result

    def checkExtension(self, name):
        """Check whether the given extension is supported by current context"""
        if not name:
            return True
        context = self.GetCurrentContext()
        if context:
            from OpenGL import contextdata
            set = contextdata.getValue('extensions', context=context)
            if set is None:
                set = {}
                contextdata.setValue('extensions', set, context=context, weak=False)
            current = set.get(name)
            if current is None:
                from OpenGL import extensions
                result = extensions.ExtensionQuerier.hasExtension(name)
                set[name] = result
                return result
            return current
        else:
            from OpenGL import extensions
            return extensions.ExtensionQuerier.hasExtension(name)
    createExtensionFunction = createBaseFunction

    def copyBaseFunction(self, original):
        """Create a new base function based on an already-created function
        
        This is normally used to provide type-specific convenience versions of
        a definition created by the automated generator.
        """
        from OpenGL import wrapper, error
        if isinstance(original, _NullFunctionPointer):
            return self.nullFunction(original.__name__, original.DLL, resultType=original.restype, argTypes=original.argtypes, doc=original.__doc__, argNames=original.argNames, extension=original.extension, deprecated=original.deprecated, error_checker=original.error_checker)
        elif hasattr(original, 'originalFunction'):
            original = original.originalFunction
        return self.createBaseFunction(original.__name__, original.DLL, resultType=original.restype, argTypes=original.argtypes, doc=original.__doc__, argNames=original.argNames, extension=original.extension, deprecated=original.deprecated, error_checker=original.errcheck)

    def nullFunction(self, functionName, dll, resultType=ctypes.c_int, argTypes=(), doc=None, argNames=(), extension=None, deprecated=False, module=None, error_checker=None, force_extension=False):
        """Construct a "null" function pointer"""
        if deprecated:
            base = _DeprecatedFunctionPointer
        else:
            base = _NullFunctionPointer
        cls = type(functionName, (base,), {'__doc__': doc, 'deprecated': deprecated})
        if MODULE_ANNOTATIONS:
            if not module:
                module = _find_module()
            if module:
                cls.__module__ = module
        return cls(functionName, dll, resultType, argTypes, argNames, extension=extension, doc=doc, error_checker=error_checker, force_extension=force_extension)

    def GetCurrentContext(self):
        """Retrieve opaque pointer for the current context"""
        raise NotImplementedError('Platform does not define a GetCurrentContext function')

    def getGLUTFontPointer(self, constant):
        """Retrieve a GLUT font pointer for this platform"""
        raise NotImplementedError('Platform does not define a GLUT font retrieval function')

    @lazy_property
    def CurrentContextIsValid(self):
        return self.GetCurrentContext

    @lazy_property
    def OpenGL(self):
        return self.GL