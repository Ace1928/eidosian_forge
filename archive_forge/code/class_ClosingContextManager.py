import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
class ClosingContextManager:

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()