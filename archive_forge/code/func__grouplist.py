import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _grouplist(self, lines):
    return [GroupInfo(*line.split()) for line in lines]