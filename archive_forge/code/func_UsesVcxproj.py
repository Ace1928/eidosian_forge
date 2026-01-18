import errno
import os
import re
import subprocess
import sys
import glob
def UsesVcxproj(self):
    """Returns true if this version uses a vcxproj file."""
    return self.uses_vcxproj