import sys
import subprocess
import py
from subprocess import Popen, PIPE
 return unicode output of executing 'cmd' in a separate process.

    raise cmdexec.Error exeception if the command failed.
    the exception will provide an 'err' attribute containing
    the error-output from the command.
    if the subprocess module does not provide a proper encoding/unicode strings
    sys.getdefaultencoding() will be used, if that does not exist, 'UTF-8'.
    