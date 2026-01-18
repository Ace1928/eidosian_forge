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
@classmethod
def _parseChanModesParam(self, params):
    """
        Parse the ISUPPORT "CHANMODES" parameter.

        See L{isupport_CHANMODES} for a detailed explanation of this parameter.
        """
    names = ('addressModes', 'param', 'setParam', 'noParam')
    if len(params) > len(names):
        raise ValueError('Expecting a maximum of %d channel mode parameters, got %d' % (len(names), len(params)))
    items = map(lambda key, value: (key, value or ''), names, params)
    return dict(items)