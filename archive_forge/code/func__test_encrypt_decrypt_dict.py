from oslo_config import cfg
from heat.common import config
from heat.common import crypt
from heat.common import exception
from heat.tests import common
def _test_encrypt_decrypt_dict(self, encryption_key=None):
    data = {'p1': u'happy', '2': [u'a', u'little', u'blue'], 'p3': {u'really': u'exited', u'ok int': 9}, '4': u'', 'p5': True, '6': 7}
    encrypted_data = crypt.encrypted_dict(data, encryption_key)
    for k in encrypted_data:
        self.assertEqual('cryptography_decrypt_v1', encrypted_data[k][0])
        self.assertEqual(2, len(encrypted_data[k]))
    self.assertEqual(set(data), set(encrypted_data))
    decrypted_data = crypt.decrypted_dict(encrypted_data, encryption_key)
    self.assertEqual(data, decrypted_data)