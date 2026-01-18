import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _run_genepop(self, extensions, option, fname, opts=None):
    if opts is None:
        opts = {}
    cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    self.controller.set_menu(option)
    if os.path.isabs(fname):
        self.controller.set_input(fname)
    else:
        self.controller.set_input(cwd + os.sep + fname)
    for opt in opts:
        self.controller.set_parameter(opt, opt + '=' + str(opts[opt]))
    self.controller()
    os.chdir(cwd)
    shutil.rmtree(temp_dir)