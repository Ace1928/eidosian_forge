import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def CreateCMakeTargetBaseName(qualified_target):
    """This is the name we would like the target to have."""
    _, gyp_target_name, gyp_target_toolset = gyp.common.ParseQualifiedTarget(qualified_target)
    cmake_target_base_name = gyp_target_name
    if gyp_target_toolset and gyp_target_toolset != 'target':
        cmake_target_base_name += '_' + gyp_target_toolset
    return StringToCMakeTargetName(cmake_target_base_name)