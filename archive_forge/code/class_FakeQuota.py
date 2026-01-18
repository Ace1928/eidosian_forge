import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeQuota(object):
    """Fake one or more Quota"""

    @staticmethod
    def create_one_quota(attrs=None):
        """Create a fake quota

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        quota_info = {'resource': 'Cluster', 'created_at': '2017-09-15T05:40:34+00:00', 'updated_at': '2017-09-15T05:40:34+00:00', 'hard_limit': 1, 'project_id': 'be24b6fba2ed4476b2bd01fa8f0ba74e', 'id': 10, 'name': 'fake-quota'}
        quota_info.update(attrs)
        quota = osc_fakes.FakeResource(info=copy.deepcopy(quota_info), loaded=True)
        return quota