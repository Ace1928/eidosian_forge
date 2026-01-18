from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import service_utils
from heat.db import api as db_api
from heat.objects import base as heat_base
@classmethod
def active_service_count(cls, context):
    """Return the number of services reportedly active."""
    return len([svc for svc in cls.get_all(context) if service_utils.format_service(svc)['status'] == 'up'])