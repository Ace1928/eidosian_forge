import os
import posixpath
import sys
import warnings
from pyarrow.util import doc, _DEPR_MSG
from pyarrow.filesystem import FileSystem
import pyarrow._hdfsio as _hdfsio
def _derive_hadoop_classpath():
    import subprocess
    find_args = ('find', '-L', os.environ['HADOOP_HOME'], '-name', '*.jar')
    find = subprocess.Popen(find_args, stdout=subprocess.PIPE)
    xargs_echo = subprocess.Popen(('xargs', 'echo'), stdin=find.stdout, stdout=subprocess.PIPE)
    jars = subprocess.check_output(('tr', "' '", "':'"), stdin=xargs_echo.stdout)
    hadoop_conf = os.environ['HADOOP_CONF_DIR'] if 'HADOOP_CONF_DIR' in os.environ else os.environ['HADOOP_HOME'] + '/etc/hadoop'
    return (hadoop_conf + ':').encode('utf-8') + jars