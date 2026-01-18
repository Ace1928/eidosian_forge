import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def _stop_tunnel(cmd):
    pexpect.run(cmd)