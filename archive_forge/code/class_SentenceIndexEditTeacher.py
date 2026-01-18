from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from .build import build
import json
import os
class SentenceIndexEditTeacher(SentenceIndexTeacher):
    """
    Index teacher where the labels are the sentences the contain the true answer.

    Some punctuation may be removed from the context and the answer for tokenization
    purposes.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        context = paragraph['context']
        question = qa['question']
        answers = ['']
        if not qa['is_impossible']:
            answers = [a['text'] for a in qa['answers']]
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace('.', '').replace('?', '').replace('!', '')
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)
        edited_sentences = self.sent_tok.tokenize(context)
        labels = []
        label_starts = []
        for sentence in edited_sentences:
            for answer in edited_answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))
                    break
        plausible = []
        if qa['is_impossible']:
            plausible = qa['plausible_answers']
        action = {'id': 'squad', 'text': context + '\n' + question, 'labels': labels, 'plausible_answers': plausible, 'episode_done': True, 'answer_starts': label_starts}
        return action