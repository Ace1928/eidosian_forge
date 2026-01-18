import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
class ProcessRunnerError(Exception):

    def __init__(self, msg, msg_type, code=None, command=None, special_case=None):
        super(ProcessRunnerError, self).__init__('{} (discovered={}): type={}, code={}, command={}'.format(msg, special_case, msg_type, code, command))
        self.msg_type = msg_type
        self.code = code
        self.command = command
        self.special_case = special_case