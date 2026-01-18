from oslotest import base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
@resource_extend.has_resource_extenders
class _DBExtender(object):

    @resource_extend.extends('ExtendedA')
    def _extend_a(self, resp, db_obj):
        pass

    @resource_extend.extends('ExtendedB')
    def _extend_b(self, resp, db_obj):
        pass