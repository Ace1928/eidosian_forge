import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
class CreateRepository(controldir.RepositoryAcquisitionPolicy):
    """A policy of creating a new repository"""

    def __init__(self, controldir, stack_on=None, stack_on_pwd=None, require_stacking=False):
        """Constructor.

        :param controldir: The controldir to create the repository on.
        :param stack_on: A location to stack on
        :param stack_on_pwd: If stack_on is relative, the location it is
            relative to.
        """
        super().__init__(stack_on, stack_on_pwd, require_stacking)
        self._controldir = controldir

    def acquire_repository(self, make_working_trees=None, shared=False, possible_transports=None):
        """Implementation of RepositoryAcquisitionPolicy.acquire_repository

        Creates the desired repository in the controldir we already have.
        """
        if possible_transports is None:
            possible_transports = []
        else:
            possible_transports = list(possible_transports)
        possible_transports.append(self._controldir.root_transport)
        stack_on = self._get_full_stack_on()
        if stack_on:
            format = self._controldir._format
            format.require_stacking(stack_on=stack_on, possible_transports=possible_transports)
            if not self._require_stacking:
                note(gettext('Using default stacking branch {0} at {1}').format(self._stack_on, self._stack_on_pwd))
        repository = self._controldir.create_repository(shared=shared)
        self._add_fallback(repository, possible_transports=possible_transports)
        if make_working_trees is not None:
            repository.set_make_working_trees(make_working_trees)
        return (repository, True)