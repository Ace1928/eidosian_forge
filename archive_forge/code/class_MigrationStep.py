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
class MigrationStep:
    from_revisions_no_deps: Tuple[str, ...]
    to_revisions_no_deps: Tuple[str, ...]
    is_upgrade: bool
    migration_fn: Any
    if TYPE_CHECKING:

        @property
        def doc(self) -> Optional[str]:
            ...

    @property
    def name(self) -> str:
        return self.migration_fn.__name__

    @classmethod
    def upgrade_from_script(cls, revision_map: RevisionMap, script: Script) -> RevisionStep:
        return RevisionStep(revision_map, script, True)

    @classmethod
    def downgrade_from_script(cls, revision_map: RevisionMap, script: Script) -> RevisionStep:
        return RevisionStep(revision_map, script, False)

    @property
    def is_downgrade(self) -> bool:
        return not self.is_upgrade

    @property
    def short_log(self) -> str:
        return '%s %s -> %s' % (self.name, util.format_as_comma(self.from_revisions_no_deps), util.format_as_comma(self.to_revisions_no_deps))

    def __str__(self):
        if self.doc:
            return '%s %s -> %s, %s' % (self.name, util.format_as_comma(self.from_revisions_no_deps), util.format_as_comma(self.to_revisions_no_deps), self.doc)
        else:
            return self.short_log