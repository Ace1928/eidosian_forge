from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
class CrudResourceManager(base.CrudManager):
    """Manager class for manipulating Identity crud_resources."""
    resource_class = CrudResource
    collection_key = 'crud_resources'
    key = 'crud_resource'

    def get(self, crud_resource):
        return super(CrudResourceManager, self).get(crud_resource_id=base.getid(crud_resource))