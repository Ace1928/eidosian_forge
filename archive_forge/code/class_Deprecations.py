import contextlib
import os
import uuid
import warnings
import fixtures
from keystoneauth1 import fixture
from keystoneauth1 import identity as ksa_identity
from keystoneauth1 import session as ksa_session
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testresources
from keystoneclient.auth import identity as ksc_identity
from keystoneclient.common import cms
from keystoneclient import session as ksc_session
from keystoneclient import utils
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
class Deprecations(fixtures.Fixture):

    def setUp(self):
        super(Deprecations, self).setUp()
        warnings.filterwarnings('error', category=DeprecationWarning, module='^keystoneclient\\.')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='^debtcollector\\.')
        self.addCleanup(warnings.resetwarnings)

    def expect_deprecations(self):
        """Call this if the test expects to call deprecated function."""
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='^keystoneclient\\.')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='^debtcollector\\.')

    @contextlib.contextmanager
    def expect_deprecations_here(self):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='^keystoneclient\\.')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='^debtcollector\\.')
        yield
        warnings.resetwarnings()
        warnings.filterwarnings('error', category=DeprecationWarning, module='^keystoneclient\\.')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='^debtcollector\\.')