import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def CreateCMakeTargetFullName(qualified_target):
    """An unambiguous name for the target."""
    gyp_file, gyp_target_name, gyp_target_toolset = gyp.common.ParseQualifiedTarget(qualified_target)
    cmake_target_full_name = gyp_file + ':' + gyp_target_name
    if gyp_target_toolset and gyp_target_toolset != 'target':
        cmake_target_full_name += '_' + gyp_target_toolset
    return StringToCMakeTargetName(cmake_target_full_name)