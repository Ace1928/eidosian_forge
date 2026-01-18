import hashlib
import os
from enum import Enum, auto
class ENV(Enum):
    """ray.util.collective environment variables."""
    NCCL_USE_MULTISTREAM = (auto(), lambda v: (v or 'True') == 'True')

    @property
    def val(self):
        """Return the output of the lambda against the system's env value."""
        _, default_fn = self.value
        return default_fn(os.getenv(self.name))