from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
class ForeignRevision(Revision):
    """A Revision from a Foreign repository. Remembers
    information about foreign revision id and mapping.

    """

    def __init__(self, foreign_revid, mapping, *args, **kwargs):
        if 'inventory_sha1' not in kwargs:
            kwargs['inventory_sha1'] = b''
        super().__init__(*args, **kwargs)
        self.foreign_revid = foreign_revid
        self.mapping = mapping