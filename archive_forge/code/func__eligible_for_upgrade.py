import collections
import math
from typing import (
from pip._vendor.resolvelib.providers import AbstractProvider
from .base import Candidate, Constraint, Requirement
from .candidates import REQUIRES_PYTHON_IDENTIFIER
from .factory import Factory
def _eligible_for_upgrade(identifier: str) -> bool:
    """Are upgrades allowed for this project?

            This checks the upgrade strategy, and whether the project was one
            that the user specified in the command line, in order to decide
            whether we should upgrade if there's a newer version available.

            (Note that we don't need access to the `--upgrade` flag, because
            an upgrade strategy of "to-satisfy-only" means that `--upgrade`
            was not specified).
            """
    if self._upgrade_strategy == 'eager':
        return True
    elif self._upgrade_strategy == 'only-if-needed':
        user_order = _get_with_identifier(self._user_requested, identifier, default=None)
        return user_order is not None
    return False