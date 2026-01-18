from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from .build import build
import copy
import json
import os
class NoEvidenceUnionTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        if not hasattr(self, 'prefix'):
            self.prefix = ''
            self.suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        qa_dir, self.evidence_dir = _path(opt)
        opt['datafile'] = os.path.join(qa_dir, self.prefix + 'noevidence-union-' + self.suffix + '.json')
        self.id = 'triviaqa'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            data = json.load(data_file)['Data']
        for datapoint in data:
            question = datapoint['Question']
            answers = [datapoint['Answer']['Value']] + sorted(list(set(datapoint['Answer']['Aliases'])))
            yield ((question, answers), True)