from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
def _extract_privileges(self, acl: str, grantee: Role) -> set[Privilege]:
    """
        Extracts a set of :class:`lazyops.libs.dbinit.data_structures.Privilege` from a Postgres ACL statement.
        The expected format is: grantee=xxxx/grantor, where grantee and grantor are roles and the "x"s indicate
        potential privilege codes. If the grantee is absent, the grantee is PUBLIC.
        :param acl: Raw acl string from Postgres in the form of grantee=xxxx/grantor.
        :param grantee: The :class:`lazyops.libs.dbinit.entities.Role` to filter to.
        :return: A set of :class:`lazyops.libs.dbinit.data_structures.Privilege` that exist in cluster, granted to the grantee.
        """
    m = re.match('(\\w*)=(\\w*)\\/(\\w*)', acl)
    if m and m[1] == grantee.name:
        raw_privileges = m[2]
        return {self._code_to_privilege(code) for code in raw_privileges}
    return set()