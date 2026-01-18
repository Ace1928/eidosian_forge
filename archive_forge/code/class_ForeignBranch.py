from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
class ForeignBranch(Branch):
    """Branch that exists in a foreign version control system."""

    def __init__(self, mapping):
        self.mapping = mapping
        super().__init__()