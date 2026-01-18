import copy
import os
from parlai.core.teachers import (
class FbformatTeacher(FbDialogTeacher):
    """
    This task simply loads the specified file: useful for quick tests without setting up
    a new task.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('-dp', '--fromfile-datapath', type=str, help='Data file')

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        opt['datafile'] = opt['fromfile_datapath']
        super().__init__(opt, shared)