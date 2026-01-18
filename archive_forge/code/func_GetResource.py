import atexit
import pkg_resources
def GetResource(name):
    """Get a resource as a string; see _Call."""
    return _Call(pkg_resources.resource_string, name)