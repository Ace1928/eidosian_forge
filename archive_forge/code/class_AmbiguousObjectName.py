from gitdb.util import to_hex_sha
class AmbiguousObjectName(ODBError):
    """Thrown if a possibly shortened name does not uniquely represent a single object
    in the database"""