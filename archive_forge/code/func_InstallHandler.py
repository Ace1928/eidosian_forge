from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import signal
import sys
from googlecloudsdk.core import log
def InstallHandler():
    """Installs the default Cloud SDK keyboard interrupt handler."""
    try:
        signal.signal(signal.SIGINT, HandleInterrupt)
    except ValueError:
        pass