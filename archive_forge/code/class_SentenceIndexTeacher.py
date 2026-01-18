from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from .build import build
import json
import os
class SentenceIndexTeacher(IndexTeacher):
    """
    Index teacher where the labels are the sentences the contain the true answer.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download('punkt')
            self.sent_tok = nltk.data.load(st_path)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        context = paragraph['context']
        question = qa['question']
        answers = []
        if not qa['is_impossible']:
            answers = [a['text'] for a in qa['answers']]
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace('.', '').replace('?', '').replace('!', '')
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)
        edited_sentences = self.sent_tok.tokenize(context)
        sentences = []
        for sentence in edited_sentences:
            for i in range(len(edited_answers)):
                sentence = sentence.replace(edited_answers[i], answers[i])
                sentences.append(sentence)
        for i in range(len(edited_answers)):
            context = context.replace(edited_answers[i], answers[i])
        labels = []
        label_starts = []
        for sentence in sentences:
            for answer in answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))
                    break
        if len(labels) == 0:
            labels.append('')
        plausible = []
        if qa['is_impossible']:
            plausible = qa['plausible_answers']
        action = {'id': 'squad', 'text': context + '\n' + question, 'labels': labels, 'plausible_answers': plausible, 'episode_done': True, 'answer_starts': label_starts}
        return action