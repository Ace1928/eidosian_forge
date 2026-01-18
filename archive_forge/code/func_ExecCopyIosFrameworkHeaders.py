import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def ExecCopyIosFrameworkHeaders(self, framework, *copy_headers):
    header_path = os.path.join(framework, 'Headers')
    if not os.path.exists(header_path):
        os.makedirs(header_path)
    for header in copy_headers:
        shutil.copy(header, os.path.join(header_path, os.path.basename(header)))