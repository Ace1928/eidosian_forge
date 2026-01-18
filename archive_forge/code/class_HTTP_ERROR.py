import logging
import socket
from subprocess import PIPE
from subprocess import Popen
import sys
import time
import traceback
import requests
from saml2test.check import CRITICAL
class HTTP_ERROR(Exception):
    pass