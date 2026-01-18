from .domreg import getDOMImplementation, registerDOMImplementation
class UserDataHandler:
    """Class giving the operation constants for UserDataHandler.handle()."""
    NODE_CLONED = 1
    NODE_IMPORTED = 2
    NODE_DELETED = 3
    NODE_RENAMED = 4