import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def StringToCMakeTargetName(a):
    """Converts the given string 'a' to a valid CMake target name.

  All invalid characters are replaced by '_'.
  Invalid for cmake: ' ', '/', '(', ')', '"'
  Invalid for make: ':'
  Invalid for unknown reasons but cause failures: '.'
  """
    return a.translate(_maketrans(' /():."', '_______'))