from oslo_config import cfg
from oslo_db import options
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context as glance_context
import glance.db.sqlalchemy.api
from glance.db.sqlalchemy import models as db_models
from glance.db.sqlalchemy import models_metadef as metadef_models
import glance.tests.functional.db as db_tests
from glance.tests.functional.db import base
from glance.tests.functional.db import base_metadef
@staticmethod
def _propdict(list_of_props):
    """
        Convert a list of ImageProperty objects to dict, ignoring
        deleted values.
        """
    return {x.name: x.value for x in list_of_props if x.deleted == 0}