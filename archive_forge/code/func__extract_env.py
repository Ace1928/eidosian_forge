import os
import re
import shutil
import sys
def _extract_env(self, arglist):
    """Extract all leading NAME=VALUE arguments from arglist."""
    envs = set()
    for arg in arglist:
        if '=' not in arg:
            break
        envs.add(arg.partition('=')[0])
    return envs