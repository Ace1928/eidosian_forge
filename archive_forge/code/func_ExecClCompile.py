import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecClCompile(self, project_dir, selected_files):
    """Executed by msvs-ninja projects when the 'ClCompile' target is used to
    build selected C/C++ files."""
    project_dir = os.path.relpath(project_dir, BASE_DIR)
    selected_files = selected_files.split(';')
    ninja_targets = [os.path.join(project_dir, filename) + '^^' for filename in selected_files]
    cmd = ['ninja.exe']
    cmd.extend(ninja_targets)
    return subprocess.call(cmd, shell=True, cwd=BASE_DIR)