import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _get_user_name_field_size(self):
    """Return the size of the user name field for the backend.

        Subclasses can override this method to indicate that the user name
        field is limited in length. The user name is the field used in the test
        that validates that a filter value works even if it's longer than a
        field.

        If the backend doesn't limit the value length then return None.

        """
    return None