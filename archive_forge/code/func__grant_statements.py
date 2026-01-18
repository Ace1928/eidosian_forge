from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
def _grant_statements(self, grantee: Role, privileges: set[Privilege]) -> Sequence[TextClause]:
    """
        Generates a grant statement to commit via SQL.
        :param grantee: The :class:`lazyops.libs.dbinit.entities.Role` to grant privileges to.
        :param privileges: The set of :class:`lazyops.libs.dbinit.data_structures.Privilege` to grant.
        :return: A Sequence of :class:`sqlalchemy.TextClause` that represent the desired grant statements.
        """
    return [text(f'GRANT {self._format_privileges(privileges)} ON {self.__class__.__name__} {self.encoded_grant_name} TO {grantee.encoded_name}')]