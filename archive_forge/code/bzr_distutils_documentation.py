import os
import re
from distutils import log
from distutils.core import Command
from distutils.dep_util import newer
from distutils.spawn import find_executable
from typing import List, Optional
Run msgfmt for each language