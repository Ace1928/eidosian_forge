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
class StampStep(MigrationStep):

    def __init__(self, from_: Optional[Union[str, Collection[str]]], to_: Optional[Union[str, Collection[str]]], is_upgrade: bool, branch_move: bool, revision_map: Optional[RevisionMap]=None) -> None:
        self.from_: Tuple[str, ...] = util.to_tuple(from_, default=())
        self.to_: Tuple[str, ...] = util.to_tuple(to_, default=())
        self.is_upgrade = is_upgrade
        self.branch_move = branch_move
        self.migration_fn = self.stamp_revision
        self.revision_map = revision_map
    doc: Optional[str] = None

    def stamp_revision(self, **kw: Any) -> None:
        return None

    def __eq__(self, other):
        return isinstance(other, StampStep) and other.from_revisions == self.from_revisions and (other.to_revisions == self.to_revisions) and (other.branch_move == self.branch_move) and (self.is_upgrade == other.is_upgrade)

    @property
    def from_revisions(self):
        return self.from_

    @property
    def to_revisions(self) -> Tuple[str, ...]:
        return self.to_

    @property
    def from_revisions_no_deps(self) -> Tuple[str, ...]:
        return self.from_

    @property
    def to_revisions_no_deps(self) -> Tuple[str, ...]:
        return self.to_

    @property
    def delete_version_num(self) -> str:
        assert len(self.from_) == 1
        return self.from_[0]

    @property
    def insert_version_num(self) -> str:
        assert len(self.to_) == 1
        return self.to_[0]

    def update_version_num(self, heads: Set[str]) -> Tuple[str, str]:
        assert len(self.from_) == 1
        assert len(self.to_) == 1
        return (self.from_[0], self.to_[0])

    def merge_branch_idents(self, heads: Union[Set[str], List[str]]) -> Union[Tuple[List[Any], str, str], Tuple[List[str], str, str]]:
        return (list(self.from_[0:-1]), self.from_[-1], self.to_[0])

    def unmerge_branch_idents(self, heads: Set[str]) -> Tuple[str, str, List[str]]:
        return (self.from_[0], self.to_[-1], list(self.to_[0:-1]))

    def should_delete_branch(self, heads: Set[str]) -> bool:
        return self.is_downgrade and self.branch_move

    def should_create_branch(self, heads: Set[str]) -> Union[Set[str], bool]:
        return self.is_upgrade and (self.branch_move or set(self.from_).difference(heads)) and set(self.to_).difference(heads)

    def should_merge_branches(self, heads: Set[str]) -> bool:
        return len(self.from_) > 1

    def should_unmerge_branches(self, heads: Set[str]) -> bool:
        return len(self.to_) > 1

    @property
    def info(self) -> MigrationInfo:
        up, down = (self.to_, self.from_) if self.is_upgrade else (self.from_, self.to_)
        assert self.revision_map is not None
        return MigrationInfo(revision_map=self.revision_map, up_revisions=up, down_revisions=down, is_upgrade=self.is_upgrade, is_stamp=True)