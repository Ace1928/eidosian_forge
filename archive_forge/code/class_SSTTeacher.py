from parlai.core.teachers import DialogTeacher
from .build import build
import os
class SSTTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        self.dt = opt['datatype'].split(':')[0]
        self.id = 'sst'
        self.SST_LABELS = ['negative', 'positive']
        opt['datafile'] = self._path(opt)
        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)
        dt = opt['datatype'].split(':')[0]
        if dt == 'valid':
            dt = 'dev'
        fname = dt + '_binary_sent.csv'
        path = os.path.join(opt['datapath'], 'SST', fname)
        return path

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.all_lines = [l.strip().split(',', 1) for l in data_file.read().split('\n')[1:-1]]
            self.labels = [self.SST_LABELS[int(x[0])] for x in self.all_lines]
            self.contexts = [x[1] for x in self.all_lines]
        new_episode = True
        self.question = 'Is this sentence positive or negative?'
        for i in range(len(self.contexts)):
            if self.labels[i]:
                yield ((self.contexts[i] + '\n' + self.question, [self.labels[i]], None, None), new_episode)

    def label_candidates(self):
        return self.SST_LABELS