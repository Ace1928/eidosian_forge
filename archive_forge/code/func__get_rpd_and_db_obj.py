from oslo_config import cfg
from heat.db import api as db_api
from heat.db import models
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests import utils
def _get_rpd_and_db_obj(self):
    rpd_obj = rpd_object.ResourcePropertiesData().create_or_update(self.ctx, self.data)
    with db_api.context_manager.reader.using(self.ctx):
        db_obj = self.ctx.session.get(models.ResourcePropertiesData, rpd_obj.id)
    self.assertEqual(len(self.data), len(db_obj['data']))
    return (rpd_obj, db_obj)