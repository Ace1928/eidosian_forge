from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def _lock_write(self, write_lockable):
    """Lock write_lockable, add a cleanup and return the result.

        :param write_lockable: An object with a lock_write method.
        :return: The result of write_lockable.lock_write().
        """
    result = write_lockable.lock_write()
    self.addCleanup(result.unlock)
    return result