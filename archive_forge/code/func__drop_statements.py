from __future__ import annotations
from typing import Sequence
from lazyops.libs.dbinit.base import Engine, TextClause, create_engine, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.cluster_entity import ClusterEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
from lazyops.libs.dbinit.mixins.grantable import Grantable
def _drop_statements(self) -> Sequence[TextClause]:
    statements = []
    if self.is_template:
        statements.append(text(f'ALTER DATABASE {self.encoded_name} is_template false'))
    statements.append(text(f'DROP DATABASE {self.encoded_name} (FORCE)'))
    return statements