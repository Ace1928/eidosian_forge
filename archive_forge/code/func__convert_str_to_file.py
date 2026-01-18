import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
def _convert_str_to_file(source, dirname):
    """Helper function to create a file ``source.c`` in `dirname` that contains
    the string in `source`. Returns the file name
    """
    filename = dirname / 'source.c'
    with filename.open('w') as f:
        f.write(str(source))
    return filename