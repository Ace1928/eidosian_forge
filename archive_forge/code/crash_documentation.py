import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
Report a bug to apport for optional automatic filing.

    :returns: The name of the crash file, or None if we didn't write one.
    