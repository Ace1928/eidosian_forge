import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _consume_stderr(self, chan):
    """
        Try to consume stderr data from chan if it's receive ready.
        """
    stderr = self._consume_data_from_channel(chan=chan, recv_method=chan.recv_stderr, recv_ready_method=chan.recv_stderr_ready)
    return stderr