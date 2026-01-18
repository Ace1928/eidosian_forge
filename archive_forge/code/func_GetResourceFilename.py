import atexit
import pkg_resources
def GetResourceFilename(name):
    """Get a filename for a resource; see _Call."""
    global _extracted_files
    if not _extracted_files:
        atexit.register(pkg_resources.cleanup_resources)
        _extracted_files = True
    return _Call(pkg_resources.resource_filename, name)