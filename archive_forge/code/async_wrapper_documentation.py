from __future__ import absolute_import, division, print_function
import errno
import json
import shlex
import shutil
import os
import subprocess
import sys
import traceback
import signal
import time
import syslog
import multiprocessing
from ansible.module_utils.common.text.converters import to_text, to_bytes

    Used to filter unrelated output around module JSON output, like messages from
    tcagetattr, or where dropbear spews MOTD on every single command (which is nuts).

    Filters leading lines before first line-starting occurrence of '{', and filter all
    trailing lines after matching close character (working from the bottom of output).
    