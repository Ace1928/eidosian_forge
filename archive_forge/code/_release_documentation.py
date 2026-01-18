import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile

        @return: A L{incremental.Version} specifying the version number of the
            project based on live python modules.
        