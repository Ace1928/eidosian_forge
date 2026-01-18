from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence
from lazyops.libs.dbinit.base import TextClause, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.exceptions import EntityExistsError, InvalidPrivilegeError
@staticmethod
def _format_privileges(privileges: set[Privilege]) -> str:
    """
        Helper method that formats privileges for a SQL statement.
        :param privileges: A set of :class:`lazyops.libs.dbinit.data_structures.Privilege` to format.
        :return: A comma-separated list of the provided privileges as a string.
        """
    return ', '.join(privileges)