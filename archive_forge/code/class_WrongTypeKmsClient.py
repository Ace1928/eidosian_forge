import pytest
from datetime import timedelta
import pyarrow as pa
class WrongTypeKmsClient:
    """This is not an implementation of KmsClient.
        """

    def __init__(self, config):
        self.master_keys_map = config.custom_kms_conf

    def wrap_key(self, key_bytes, master_key_identifier):
        return None

    def unwrap_key(self, wrapped_key, master_key_identifier):
        return None