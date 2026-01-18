from io import BytesIO
from .. import osutils
class TreeSupplement:
    """Supplement for a Bazaar tree roundtripped into Git.

    This provides file ids (if they are different from the mapping default)
    and can provide text revisions.
    """