import os
import sys
import threading
from . import process
from . import reduction
class SpawnContext(BaseContext):
    _name = 'spawn'
    Process = SpawnProcess