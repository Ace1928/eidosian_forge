from gitdb.util import to_hex_sha
class InvalidDBRoot(ODBError):
    """Thrown if an object database cannot be initialized at the given path"""