import functools
import logging
from multiprocessing import managers
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from oslo_rootwrap import cmd
from oslo_rootwrap import jsonrpc
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
class RootwrapManager(managers.BaseManager):

    def __init__(self, address=None, authkey=None):
        super(RootwrapManager, self).__init__(address, authkey, serializer='jsonrpc')