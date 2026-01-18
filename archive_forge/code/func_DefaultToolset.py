import errno
import os
import re
import subprocess
import sys
import glob
def DefaultToolset(self):
    """Returns the msbuild toolset version that will be used in the absence
    of a user override."""
    return self.default_toolset