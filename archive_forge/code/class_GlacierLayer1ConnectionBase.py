import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
class GlacierLayer1ConnectionBase(AWSMockServiceTestCase):
    connection_class = Layer1

    def setUp(self):
        super(GlacierLayer1ConnectionBase, self).setUp()
        self.json_header = [('Content-Type', 'application/json')]
        self.vault_name = u'examplevault'
        self.vault_arn = 'arn:aws:glacier:us-east-1:012345678901:vaults/' + self.vault_name
        self.vault_info = {u'CreationDate': u'2012-03-16T22:22:47.214Z', u'LastInventoryDate': u'2012-03-21T22:06:51.218Z', u'NumberOfArchives': 2, u'SizeInBytes': 12334, u'VaultARN': self.vault_arn, u'VaultName': self.vault_name}