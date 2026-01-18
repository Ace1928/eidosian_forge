import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _rendezvous_error(msg):
    return ValueError('Error initializing torch.distributed using ' + msg)