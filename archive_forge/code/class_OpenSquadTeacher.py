from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from .build import build
import json
import os
class OpenSquadTeacher(DialogTeacher):
    """
    This version of SQuAD inherits from the core Dialog Teacher, which just requires it
    to define an iterator over its data `setup_data` in order to inherit basic metrics,
    a default `act` function.

    Note: This teacher omits the context paragraph
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        opt['datafile'] = os.path.join(opt['datapath'], 'SQuAD2', suffix + '-v2.0.json')
        self.id = 'squad2'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        for article in self.squad:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    question = qa['question']
                    ans_iter = [{'text': ''}]
                    if not qa['is_impossible']:
                        ans_iter = qa['answers']
                    answers = (a['text'] for a in ans_iter)
                    yield ((question, answers), True)