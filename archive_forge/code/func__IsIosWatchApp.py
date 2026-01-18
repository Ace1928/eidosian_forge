import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _IsIosWatchApp(self):
    return int(self.spec.get('ios_watch_app', 0)) != 0