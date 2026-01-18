from unittest import mock
from heat.common import exception
from heat.db import api as db_api
from heat.tests import utils
class RealityStore(object):

    def __init__(self):
        self.cntxt = utils.dummy_context()

    def resources_by_logical_name(self, logical_name):
        ret = []
        resources = db_api.resource_get_all(self.cntxt)
        for res in resources:
            if res.name == logical_name and res.action in ('CREATE', 'UPDATE') and (res.status == 'COMPLETE'):
                ret.append(res)
        return ret

    def all_resources(self):
        try:
            resources = db_api.resource_get_all(self.cntxt)
        except exception.NotFound:
            return []
        ret = []
        for res in resources:
            if res.action in ('CREATE', 'UPDATE') and res.status == 'COMPLETE':
                ret.append(res)
        return ret

    def resource_properties(self, res, prop_name):
        res_data = db_api.resource_data_get_by_key(self.cntxt, res.id, prop_name)
        return res_data.value