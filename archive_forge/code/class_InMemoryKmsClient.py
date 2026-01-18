import base64
import pyarrow.parquet.encryption as pe
class InMemoryKmsClient(pe.KmsClient):
    """This is a mock class implementation of KmsClient, built for testing
    only.
    """

    def __init__(self, config):
        """Create an InMemoryKmsClient instance."""
        pe.KmsClient.__init__(self)
        self.master_keys_map = config.custom_kms_conf

    def wrap_key(self, key_bytes, master_key_identifier):
        """Not a secure cipher - the wrapped key
        is just the master key concatenated with key bytes"""
        master_key_bytes = self.master_keys_map[master_key_identifier].encode('utf-8')
        wrapped_key = b''.join([master_key_bytes, key_bytes])
        result = base64.b64encode(wrapped_key)
        return result

    def unwrap_key(self, wrapped_key, master_key_identifier):
        """Not a secure cipher - just extract the key from
        the wrapped key"""
        expected_master_key = self.master_keys_map[master_key_identifier]
        decoded_wrapped_key = base64.b64decode(wrapped_key)
        master_key_bytes = decoded_wrapped_key[:16]
        decrypted_key = decoded_wrapped_key[16:]
        if expected_master_key == master_key_bytes.decode('utf-8'):
            return decrypted_key
        raise ValueError('Incorrect master key used', master_key_bytes, decrypted_key)