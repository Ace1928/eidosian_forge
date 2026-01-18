import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def CalculateGeneratorInputInfo(params):
    """Calculate the generator specific info that gets fed to input (called by
    gyp)."""
    generator_flags = params.get('generator_flags', {})
    android_ndk_version = generator_flags.get('android_ndk_version', None)
    if android_ndk_version:
        global generator_wants_sorted_dependencies
        generator_wants_sorted_dependencies = True
    output_dir = params['options'].generator_output or params['options'].toplevel_dir
    builddir_name = generator_flags.get('output_dir', 'out')
    qualified_out_dir = os.path.normpath(os.path.join(output_dir, builddir_name, 'gypfiles'))
    global generator_filelist_paths
    generator_filelist_paths = {'toplevel': params['options'].toplevel_dir, 'qualified_out_dir': qualified_out_dir}