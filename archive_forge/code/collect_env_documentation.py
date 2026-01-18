import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages.