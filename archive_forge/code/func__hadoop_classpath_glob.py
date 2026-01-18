import os
import posixpath
import sys
import warnings
from pyarrow.util import doc, _DEPR_MSG
from pyarrow.filesystem import FileSystem
import pyarrow._hdfsio as _hdfsio
def _hadoop_classpath_glob(hadoop_bin):
    import subprocess
    hadoop_classpath_args = (hadoop_bin, 'classpath', '--glob')
    return subprocess.check_output(hadoop_classpath_args)