import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class DccSendFactory(protocol.Factory):
    protocol = DccSendProtocol

    def __init__(self, file):
        self.file = file

    def buildProtocol(self, connection):
        p = self.protocol(self.file)
        p.factory = self
        return p