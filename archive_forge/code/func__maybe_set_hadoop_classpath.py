import os
import posixpath
import sys
import warnings
from pyarrow.util import doc, _DEPR_MSG
from pyarrow.filesystem import FileSystem
import pyarrow._hdfsio as _hdfsio
def _maybe_set_hadoop_classpath():
    import re
    if re.search('hadoop-common[^/]+.jar', os.environ.get('CLASSPATH', '')):
        return
    if 'HADOOP_HOME' in os.environ:
        if sys.platform != 'win32':
            classpath = _derive_hadoop_classpath()
        else:
            hadoop_bin = '{}/bin/hadoop'.format(os.environ['HADOOP_HOME'])
            classpath = _hadoop_classpath_glob(hadoop_bin)
    else:
        classpath = _hadoop_classpath_glob('hadoop')
    os.environ['CLASSPATH'] = classpath.decode('utf-8')