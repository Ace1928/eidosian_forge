import os
import re
import shutil
import sys
def _getuid(user):
    """Return uid for user."""
    return pwd.getpwnam(user).pw_uid