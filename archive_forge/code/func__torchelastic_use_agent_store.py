import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _torchelastic_use_agent_store() -> bool:
    return os.environ.get('TORCHELASTIC_USE_AGENT_STORE', None) == str(True)