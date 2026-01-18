import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
def _make_source(name, init, body):
    """ Combines the code fragments into source code ready to be compiled
    """
    code = '\n    #include <Python.h>\n\n    %(body)s\n\n    PyMODINIT_FUNC\n    PyInit_%(name)s(void) {\n    %(init)s\n    }\n    ' % dict(name=name, init=init, body=body)
    return code