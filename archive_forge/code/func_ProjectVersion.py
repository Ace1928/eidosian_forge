import errno
import os
import re
import subprocess
import sys
import glob
def ProjectVersion(self):
    """Get the version number of the vcproj or vcxproj files."""
    return self.project_version