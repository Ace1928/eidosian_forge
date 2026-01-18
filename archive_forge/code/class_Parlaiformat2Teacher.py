import copy
import os
from parlai.core.teachers import (
class Parlaiformat2Teacher(ParlAIDialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath2', type=str, help='Data file')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath2'):
            raise RuntimeError('fromfile_datapath2 not specified')
        datafile = opt['fromfile_datapath2']
        if shared is None:
            self._setup_data(datafile)
        self.id = datafile
        self.reset()