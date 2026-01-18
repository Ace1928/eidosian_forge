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
def _consume_stdout(self, chan):
    """
        Try to consume stdout data from chan if it's receive ready.
        """
    stdout = self._consume_data_from_channel(chan=chan, recv_method=chan.recv, recv_ready_method=chan.recv_ready)
    return stdout