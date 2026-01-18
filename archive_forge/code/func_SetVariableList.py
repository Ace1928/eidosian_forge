import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def SetVariableList(output, variable_name, values):
    """Sets a CMake variable to a list."""
    if not values:
        return SetVariable(output, variable_name, '')
    if len(values) == 1:
        return SetVariable(output, variable_name, values[0])
    output.write('list(APPEND ')
    output.write(variable_name)
    output.write('\n  "')
    output.write('"\n  "'.join([CMakeStringEscape(value) for value in values]))
    output.write('")\n')