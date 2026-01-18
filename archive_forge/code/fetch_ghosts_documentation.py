import contextlib
from .branch import Branch
from .errors import CommandError, NoSuchRevision
from .trace import note
Find all ancestors that aren't stored in this branch.