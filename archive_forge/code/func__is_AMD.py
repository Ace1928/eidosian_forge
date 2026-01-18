import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_AMD(self):
    return self.info[0]['VendorIdentifier'] == 'AuthenticAMD'