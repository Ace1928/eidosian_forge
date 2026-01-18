import errno
import os
import re
import subprocess
import sys
import glob
def ProjectExtension(self):
    """Returns the file extension for the project."""
    return self.uses_vcxproj and '.vcxproj' or '.vcproj'