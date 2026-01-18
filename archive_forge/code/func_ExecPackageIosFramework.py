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
def ExecPackageIosFramework(self, framework):
    binary = os.path.basename(framework).split('.')[0]
    module_path = os.path.join(framework, 'Modules')
    if not os.path.exists(module_path):
        os.mkdir(module_path)
    module_template = 'framework module %s {\n  umbrella header "%s.h"\n\n  export *\n  module * { export * }\n}\n' % (binary, binary)
    with open(os.path.join(module_path, 'module.modulemap'), 'w') as module_file:
        module_file.write(module_template)