from __future__ import annotations
from contextlib import contextmanager
from contextlib import nullcontext
import logging
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.engine import url as sqla_url
from sqlalchemy.engine.strategies import MockEngineStrategy
from .. import ddl
from .. import util
from ..util import sqla_compat
from ..util.compat import EncodedIO
def _unmerge_to_revisions(self, heads: Set[str]) -> Tuple[str, ...]:
    other_heads = set(heads).difference([self.revision.revision])
    if other_heads:
        ancestors = {r.revision for r in self.revision_map._get_ancestor_nodes(self.revision_map.get_revisions(other_heads), check=False)}
        return tuple(set(self.to_revisions).difference(ancestors))
    else:
        ancestors = {r.revision for to_revision in self.to_revisions for r in self.revision_map._get_ancestor_nodes(self.revision_map.get_revisions(to_revision), check=False) if r.revision != to_revision}
        return tuple(set(self.to_revisions).difference(ancestors))