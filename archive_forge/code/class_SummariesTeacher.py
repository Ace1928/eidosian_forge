from parlai.core.teachers import DialogTeacher
from .build import build
import os
import copy
import csv
import glob
class SummariesTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'NarrativeQA'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading data from: ' + path)
        qa_path = os.path.join(path, 'qaps.csv')
        summaries_path = os.path.join(path, 'summaries.csv')
        qa_pairs = dict()
        with open(qa_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['document_id'] not in qa_pairs:
                    qa_pairs[row['document_id']] = []
                qa_pairs[row['document_id']].append(row)
        with open(summaries_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                info = 'Summary:  %s' % row['summary_tokenized']
                for i, qa in enumerate(qa_pairs[row['document_id']]):
                    question = qa['question_tokenized']
                    answer1 = qa['answer1_tokenized']
                    answer2 = qa['answer2_tokenized']
                    if i == 0:
                        yield ((info + '\n' + question, [answer1, answer2]), True)
                    else:
                        yield ((question, [answer1, answer2]), False)