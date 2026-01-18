from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.db import api as db_api
from heat.objects import base as heat_base
class StackTagList(heat_base.HeatObject, base.ObjectListBase):
    fields = {'objects': fields.ListOfObjectsField('StackTag')}

    def __init__(self, *args, **kwargs):
        self._changed_fields = set()
        super(StackTagList, self).__init__()

    @classmethod
    def get(cls, context, stack_id):
        db_tags = db_api.stack_tags_get(context, stack_id)
        if db_tags:
            return base.obj_make_list(context, cls(), StackTag, db_tags)

    @classmethod
    def set(cls, context, stack_id, tags):
        db_tags = db_api.stack_tags_set(context, stack_id, tags)
        if db_tags:
            return base.obj_make_list(context, cls(), StackTag, db_tags)

    @classmethod
    def delete(cls, context, stack_id):
        db_api.stack_tags_delete(context, stack_id)

    @classmethod
    def from_db_object(cls, context, db_tags):
        if db_tags is not None:
            return base.obj_make_list(context, cls(), StackTag, db_tags)