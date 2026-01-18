import warnings
class NamespaceNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the namespace doesn't exist."""