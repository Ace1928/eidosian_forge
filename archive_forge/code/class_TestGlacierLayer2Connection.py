from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
class TestGlacierLayer2Connection(GlacierLayer2Base):

    def setUp(self):
        GlacierLayer2Base.setUp(self)
        self.layer2 = Layer2(layer1=self.mock_layer1)

    def test_create_vault(self):
        self.mock_layer1.describe_vault.return_value = FIXTURE_VAULT
        self.layer2.create_vault('My Vault')
        self.mock_layer1.create_vault.assert_called_with('My Vault')

    def test_get_vault(self):
        self.mock_layer1.describe_vault.return_value = FIXTURE_VAULT
        vault = self.layer2.get_vault('examplevault')
        self.assertEqual(vault.layer1, self.mock_layer1)
        self.assertEqual(vault.name, 'examplevault')
        self.assertEqual(vault.size, 78088912)
        self.assertEqual(vault.number_of_archives, 192)

    def test_list_vaults(self):
        self.mock_layer1.list_vaults.return_value = FIXTURE_VAULTS
        vaults = self.layer2.list_vaults()
        self.assertEqual(vaults[0].name, 'vault0')
        self.assertEqual(len(vaults), 2)

    def test_list_vaults_paginated(self):
        resps = [FIXTURE_PAGINATED_VAULTS, FIXTURE_PAGINATED_VAULTS_CONT]

        def return_paginated_vaults_resp(marker=None, limit=None):
            return resps.pop(0)
        self.mock_layer1.list_vaults = Mock(side_effect=return_paginated_vaults_resp)
        vaults = self.layer2.list_vaults()
        self.assertEqual(vaults[0].name, 'vault0')
        self.assertEqual(vaults[3].name, 'vault3')
        self.assertEqual(len(vaults), 4)