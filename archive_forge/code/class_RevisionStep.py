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
class RevisionStep(MigrationStep):

    def __init__(self, revision_map: RevisionMap, revision: Script, is_upgrade: bool) -> None:
        self.revision_map = revision_map
        self.revision = revision
        self.is_upgrade = is_upgrade
        if is_upgrade:
            self.migration_fn = revision.module.upgrade
        else:
            self.migration_fn = revision.module.downgrade

    def __repr__(self):
        return 'RevisionStep(%r, is_upgrade=%r)' % (self.revision.revision, self.is_upgrade)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RevisionStep) and other.revision == self.revision and (self.is_upgrade == other.is_upgrade)

    @property
    def doc(self) -> Optional[str]:
        return self.revision.doc

    @property
    def from_revisions(self) -> Tuple[str, ...]:
        if self.is_upgrade:
            return self.revision._normalized_down_revisions
        else:
            return (self.revision.revision,)

    @property
    def from_revisions_no_deps(self) -> Tuple[str, ...]:
        if self.is_upgrade:
            return self.revision._versioned_down_revisions
        else:
            return (self.revision.revision,)

    @property
    def to_revisions(self) -> Tuple[str, ...]:
        if self.is_upgrade:
            return (self.revision.revision,)
        else:
            return self.revision._normalized_down_revisions

    @property
    def to_revisions_no_deps(self) -> Tuple[str, ...]:
        if self.is_upgrade:
            return (self.revision.revision,)
        else:
            return self.revision._versioned_down_revisions

    @property
    def _has_scalar_down_revision(self) -> bool:
        return len(self.revision._normalized_down_revisions) == 1

    def should_delete_branch(self, heads: Set[str]) -> bool:
        """A delete is when we are a. in a downgrade and b.
        we are going to the "base" or we are going to a version that
        is implied as a dependency on another version that is remaining.

        """
        if not self.is_downgrade:
            return False
        if self.revision.revision not in heads:
            return False
        downrevs = self.revision._normalized_down_revisions
        if not downrevs:
            return True
        else:
            to_revisions = self._unmerge_to_revisions(heads)
            return not to_revisions

    def merge_branch_idents(self, heads: Set[str]) -> Tuple[List[str], str, str]:
        other_heads = set(heads).difference(self.from_revisions)
        if other_heads:
            ancestors = {r.revision for r in self.revision_map._get_ancestor_nodes(self.revision_map.get_revisions(other_heads), check=False)}
            from_revisions = list(set(self.from_revisions).difference(ancestors))
        else:
            from_revisions = list(self.from_revisions)
        return (list(from_revisions[0:-1]), from_revisions[-1], self.to_revisions[0])

    def _unmerge_to_revisions(self, heads: Set[str]) -> Tuple[str, ...]:
        other_heads = set(heads).difference([self.revision.revision])
        if other_heads:
            ancestors = {r.revision for r in self.revision_map._get_ancestor_nodes(self.revision_map.get_revisions(other_heads), check=False)}
            return tuple(set(self.to_revisions).difference(ancestors))
        else:
            ancestors = {r.revision for to_revision in self.to_revisions for r in self.revision_map._get_ancestor_nodes(self.revision_map.get_revisions(to_revision), check=False) if r.revision != to_revision}
            return tuple(set(self.to_revisions).difference(ancestors))

    def unmerge_branch_idents(self, heads: Set[str]) -> Tuple[str, str, Tuple[str, ...]]:
        to_revisions = self._unmerge_to_revisions(heads)
        return (self.from_revisions[0], to_revisions[-1], to_revisions[0:-1])

    def should_create_branch(self, heads: Set[str]) -> bool:
        if not self.is_upgrade:
            return False
        downrevs = self.revision._normalized_down_revisions
        if not downrevs:
            return True
        elif not heads.intersection(downrevs):
            return True
        else:
            return False

    def should_merge_branches(self, heads: Set[str]) -> bool:
        if not self.is_upgrade:
            return False
        downrevs = self.revision._normalized_down_revisions
        if len(downrevs) > 1 and len(heads.intersection(downrevs)) > 1:
            return True
        return False

    def should_unmerge_branches(self, heads: Set[str]) -> bool:
        if not self.is_downgrade:
            return False
        downrevs = self.revision._normalized_down_revisions
        if self.revision.revision in heads and len(downrevs) > 1:
            return True
        return False

    def update_version_num(self, heads: Set[str]) -> Tuple[str, str]:
        if not self._has_scalar_down_revision:
            downrev = heads.intersection(self.revision._normalized_down_revisions)
            assert len(downrev) == 1, "Can't do an UPDATE because downrevision is ambiguous"
            down_revision = list(downrev)[0]
        else:
            down_revision = self.revision._normalized_down_revisions[0]
        if self.is_upgrade:
            return (down_revision, self.revision.revision)
        else:
            return (self.revision.revision, down_revision)

    @property
    def delete_version_num(self) -> str:
        return self.revision.revision

    @property
    def insert_version_num(self) -> str:
        return self.revision.revision

    @property
    def info(self) -> MigrationInfo:
        return MigrationInfo(revision_map=self.revision_map, up_revisions=self.revision.revision, down_revisions=self.revision._normalized_down_revisions, is_upgrade=self.is_upgrade, is_stamp=False)