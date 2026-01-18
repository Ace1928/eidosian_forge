import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
def add_binary_annotation(self, bannotation):
    if bannotation.host is None:
        bannotation.host = self.endpoint
    if not self._done:
        self.bannotations.append(bannotation)