import io
import os
from .context import reduction, set_spawning_popen
from . import forkserver
from . import popen_fork
from . import spawn
from . import util
class _DupFd(object):

    def __init__(self, ind):
        self.ind = ind

    def detach(self):
        return forkserver.get_inherited_fds()[self.ind]