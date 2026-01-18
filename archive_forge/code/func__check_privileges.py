from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
def _check_privileges(self, declared_privileges: set[Privilege], existing_privileges: set[Privilege]) -> bool:
    """
        Check the in-code declared privileges against the in-cluster existing privileges.
        :param declared_privileges: A set of :class:`lazyops.libs.dbinit.data_structures.Privilege` declared in code for this entity.
        :param existing_privileges: A set of :class:`lazyops.libs.dbinit.data_structures.Privilege` declared in cluster for this entity.
        :return: True if the declared privileges are a subset of the existing privileges. Accounts for ALL_PRIVILEGES.
        """
    if Privilege.ALL_PRIVILEGES in declared_privileges:
        declared = self._allowed_privileges()
        declared.discard(Privilege.ALL_PRIVILEGES)
    else:
        declared = declared_privileges
    return declared.issubset(existing_privileges)