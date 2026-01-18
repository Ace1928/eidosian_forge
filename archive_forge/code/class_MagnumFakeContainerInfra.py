import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class MagnumFakeContainerInfra(object):

    def __init__(self):
        self.cluster_templates = FakeBaseModelManager()
        self.clusters = FakeBaseModelManager()
        self.mservices = FakeBaseModelManager()
        self.certificates = FakeCertificatesModelManager()
        self.stats = FakeStatsModelManager()
        self.quotas = FakeQuotasModelManager()
        self.nodegroups = FakeNodeGroupManager()