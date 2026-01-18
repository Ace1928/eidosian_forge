import errno
import os
import re
import subprocess
import sys
import glob
def SolutionVersion(self):
    """Get the version number of the sln files."""
    return self.solution_version