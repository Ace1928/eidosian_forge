from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from .build import build
import json
import os
class IndexTeacher(FixedDialogTeacher):
    """
    Hand-written SQuAD teacher, which loads the json squad data and implements its own
    `act()` method for interacting with student agent, rather than inheriting from the
    core Dialog Teacher. This code is here as an example of rolling your own without
    inheritance.

    This teacher also provides access to the "answer_start" indices that specify the
    location of the answer in the context.
    """

    def __init__(self, opt, shared=None):
        build(opt)
        super().__init__(opt, shared)
        if self.datatype.startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        datapath = os.path.join(opt['datapath'], 'SQuAD2', suffix + '-v2.0.json')
        self.data = self._setup_data(datapath)
        self.id = 'squad2'
        self.reset()

    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        question = qa['question']
        answers = []
        answer_starts = []
        if not qa['is_impossible']:
            for a in qa['answers']:
                answers.append(a['text'])
                answer_starts.append(a['answer_start'])
        context = paragraph['context']
        plausible = qa.get('plausible_answers', [])
        action = {'id': 'squad', 'text': context + '\n' + question, 'labels': answers, 'plausible_answers': plausible, 'episode_done': True, 'answer_starts': answer_starts}
        return action

    def _setup_data(self, path):
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        self.examples = []
        for article_idx in range(len(self.squad)):
            article = self.squad[article_idx]
            for paragraph_idx in range(len(article['paragraphs'])):
                paragraph = article['paragraphs'][paragraph_idx]
                num_questions = len(paragraph['qas'])
                for qa_idx in range(num_questions):
                    self.examples.append((article_idx, paragraph_idx, qa_idx))