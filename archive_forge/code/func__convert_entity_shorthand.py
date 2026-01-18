from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _convert_entity_shorthand(self, key, shorthand_data, reference_data):
    """Convert a shorthand entity description into a full ID reference.

        In test plan definitions, we allow a shorthand for referencing to an
        entity of the form:

        'user': 0

        which is actually shorthand for:

        'user_id': reference_data['users'][0]['id']

        This method converts the shorthand version into the full reference.

        """
    expanded_key = '%s_id' % key
    reference_index = '%ss' % key
    index_value = reference_data[reference_index][shorthand_data[key]]['id']
    return (expanded_key, index_value)