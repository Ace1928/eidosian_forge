import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeQuotasModelManager(object):

    def get(self, id, resource):
        pass

    def create(self, **kwargs):
        pass

    def delete(self, id):
        pass