import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def ExpandSpecial(self, path, product_dir=None):
    """Expand specials like $!PRODUCT_DIR in |path|.

        If |product_dir| is None, assumes the cwd is already the product
        dir.  Otherwise, |product_dir| is the relative path to the product
        dir.
        """
    PRODUCT_DIR = '$!PRODUCT_DIR'
    if PRODUCT_DIR in path:
        if product_dir:
            path = path.replace(PRODUCT_DIR, product_dir)
        else:
            path = path.replace(PRODUCT_DIR + '/', '')
            path = path.replace(PRODUCT_DIR + '\\', '')
            path = path.replace(PRODUCT_DIR, '.')
    INTERMEDIATE_DIR = '$!INTERMEDIATE_DIR'
    if INTERMEDIATE_DIR in path:
        int_dir = self.GypPathToUniqueOutput('gen')
        path = path.replace(INTERMEDIATE_DIR, os.path.join(product_dir or '', int_dir))
    CONFIGURATION_NAME = '$|CONFIGURATION_NAME'
    path = path.replace(CONFIGURATION_NAME, self.config_name)
    return path