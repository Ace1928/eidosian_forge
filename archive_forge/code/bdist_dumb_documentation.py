import os
from distutils.core import Command
from distutils.util import get_platform
from distutils.dir_util import remove_tree, ensure_relative
from distutils.errors import *
from distutils.sysconfig import get_python_version
from distutils import log
distutils.command.bdist_dumb

Implements the Distutils 'bdist_dumb' command (create a "dumb" built
distribution -- i.e., just an archive to be unpacked under $prefix or
$exec_prefix).