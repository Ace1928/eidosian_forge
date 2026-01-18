import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def UnsetVariable(output, variable_name):
    """Unsets a CMake variable."""
    output.write('unset(')
    output.write(variable_name)
    output.write(')\n')