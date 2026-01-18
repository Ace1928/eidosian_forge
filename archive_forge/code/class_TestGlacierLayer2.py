import time
from tests.unit import unittest
from boto.glacier.layer2 import Layer1, Layer2
class TestGlacierLayer2(unittest.TestCase):
    glacier = True

    def setUp(self):
        self.layer2 = Layer2()
        self.vault_name = 'testvault%s' % int(time.time())

    def test_create_delete_vault(self):
        vault = self.layer2.create_vault(self.vault_name)
        retrieved_vault = self.layer2.get_vault(self.vault_name)
        self.layer2.delete_vault(self.vault_name)
        self.assertEqual(vault.name, retrieved_vault.name)
        self.assertEqual(vault.arn, retrieved_vault.arn)
        self.assertEqual(vault.creation_date, retrieved_vault.creation_date)
        self.assertEqual(vault.last_inventory_date, retrieved_vault.last_inventory_date)
        self.assertEqual(vault.number_of_archives, retrieved_vault.number_of_archives)