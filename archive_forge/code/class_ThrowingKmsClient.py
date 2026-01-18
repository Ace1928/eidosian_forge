import pytest
from datetime import timedelta
import pyarrow as pa
class ThrowingKmsClient(pe.KmsClient):
    """A KmsClient implementation that throws exception in
        wrap/unwrap calls
        """

    def __init__(self, config):
        """Create an InMemoryKmsClient instance."""
        pe.KmsClient.__init__(self)
        self.config = config

    def wrap_key(self, key_bytes, master_key_identifier):
        raise ValueError('Cannot Wrap Key')

    def unwrap_key(self, wrapped_key, master_key_identifier):
        raise ValueError('Cannot Unwrap Key')