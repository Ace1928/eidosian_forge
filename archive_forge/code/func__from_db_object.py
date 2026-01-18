from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
@staticmethod
def _from_db_object(context, sdata, db_sdata):
    if db_sdata is None:
        return None
    for field in sdata.fields:
        sdata[field] = db_sdata[field]
    sdata._context = context
    sdata.obj_reset_changes()
    return sdata