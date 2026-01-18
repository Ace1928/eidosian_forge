import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class FighterAceServerOptions(usage.Options):
    """
    Options for FighterAce 'server' subcommand
    """
    optFlags = [['list-server', None, 'List this server with the online FighterAce network']]
    optParameters = [['packets-per-second', None, 'Number of update packets to send per second', '20']]