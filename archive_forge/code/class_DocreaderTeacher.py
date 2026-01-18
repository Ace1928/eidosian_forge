import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
class DocreaderTeacher(WizardOfWikipediaTeacher):
    """
    Teacher for training a doc reader. One can specify the format of the action via the
    `--teacher-type` flag.

    docs:
        {
            text: <Passage> 
 <Sentence for which passage was retrieved>
            labels: <Sentence chosen from passage>
        }

    docs_sentence:
        {
            text: <Sentence for which passage was retrieved>
            label: <Sentence chosen from passages>
            label_candidates: <All sentences in retrieved passage>
        }

    more_docs:
        {
            text: <All retrieved passages> 

                  <Chosen topic + Last thing wizard said + last thing apprentice said>
            labels: <Sentence chosen from passages>
        }

    more_docs_sentence:
        {
            text: <Sentence for which passage was retrieved>
            label: <Sentence chosen from passages>
            label_candidates: <All sentences in all retrieved passages>
        }
    span:
        {
            text: <Sentence for which passage was retrieved>
            label: <Max overlap span between sentence said and sentence retrieved>
        }
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.num_exs = 0
        for ep in range(self.num_episodes()):
            d = self.data[ep]
            for entry in d['dialog']:
                if entry.get('checked_sentence', None) is not None and entry.get('checked_sentence') != {} and (TOKEN_NOCHOSEN not in entry.get('checked_sentence')):
                    self.num_exs += 1
        self.stop_words = ['i', 'a', 'an', 'am', 'are', 'about', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', '--', '?', '.', "''", "''", '``', ',', 'do', 'see', 'want', 'people', 'and', "n't", 'me', 'too', 'own', 'their', '*', "'s", 'not', 'than', 'other', 'you', 'your', 'know', 'just', 'but', 'does', 'really', 'have', 'into', 'more', 'also', 'has', 'any', 'why', 'will', 'with', 'well', 'still', 'he', 'she', 'we', 'may', 'these', 'his', 'hers', 'which', 'such', 'they', 'its', 'were', 'my', 'there', ';', '-', ':', '|', '&', ')', '(']
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
        self.teacher_type = opt.get('teacher_type')

    @staticmethod
    def add_cmdline_args(argparser):
        WizardDialogKnowledgeTeacher.add_cmdline_args(argparser)
        argparser.add_argument('--teacher-type', type=str, default='docs', help='determines what the action dict looks like; see docstring for examples', choices=['docs', 'docs_sentence', 'more_docs', 'more_docs_sentence', 'span_teacher'])

    def get_min_stopwords(self, word_set):
        min_count = 1000000000000
        min_words = ''
        for words in word_set:
            count = 0
            for stop in self.stop_words:
                if stop in words:
                    count += 1
            if count < min_count:
                min_count = count
                min_words = words
        return min_words

    def space_punctuation(self, words, unspace=False):
        puncs = [('.', ' .'), (',', ' ,'), ('?', ' ?'), (' !', '!'), ('(', ' ('), (')', ' )')]
        new_words = words
        for punc in puncs:
            if unspace:
                new_words = new_words.replace(punc[1], punc[0])
            else:
                new_words = new_words.replace(punc[0], punc[1])
        return new_words

    def get_span(self, one, two):
        if not one or not two:
            return None
        one_space = self.space_punctuation(one)
        two_space = self.space_punctuation(two)
        first = one_space.split(' ')
        second = two_space.split(' ')
        length = min(len(first), len(second))
        overlap = set.intersection(set(first), set(second))
        if not overlap:
            return ''
        max_span = self.space_punctuation(self.get_min_stopwords(overlap), unspace=True)
        for i in range(1, length):
            t_1 = []
            t_2 = []
            for j in range(len(first) - i):
                temp_1 = ' '.join([first[k] for k in range(j, j + i + 1)])
                t_1.append(temp_1)
            for j in range(len(second) - i):
                temp_2 = ' '.join([second[k] for k in range(j, j + i + 1)])
                t_2.append(temp_2)
            overlap = set.intersection(set(t_1), set(t_2))
            if not overlap:
                return max_span
            max_span = self.space_punctuation(self.get_min_stopwords(overlap), unspace=True)
        return max_span

    def num_examples(self):
        return self.num_exs

    def length_episode(self, dialog):
        len_ep = 0
        idxs = []
        i = 0
        for entry in dialog['dialog']:
            if entry.get('checked_sentence', None) is not None and entry.get('checked_sentence') != {} and (TOKEN_NOCHOSEN not in entry.get('checked_sentence')):
                len_ep += 1
                idxs.append(i)
            i += 1
        return (len_ep, idxs)

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        len_ep, idxs = self.length_episode(d)
        idx = idxs[entry_idx]
        episode_done = entry_idx == len_ep - 1
        checked_sentence_dict = d['dialog'][idx]['checked_sentence']
        sentence = _first_val(checked_sentence_dict)
        passage, text = self.extract_passage_and_text(d, idx)
        passages, texts = self.extract_passages_and_texts(d, idx)
        span_label = self.get_span_label(d, idx)
        action = {'id': 'WizardDocReader:{}'.format(self.teacher_type), 'labels': [sentence], 'episode_done': episode_done}
        if self.teacher_type == 'docs':
            action['text'] = '{}\n{}'.format(passage, text)
        elif self.teacher_type == 'docs_sentence':
            action['text'] = text
            action['label_candidates'] = self.sent_tok.tokenize(passage)
        elif self.teacher_type == 'more_docs':
            action['text'] = '{}\n{}'.format(passages, texts)
        elif self.teacher_type == 'more_docs_sentence':
            action['text'] = texts
            action['label_candidates'] = self.sent_tok.tokenize(passages)
            label = action['labels'][0]
            if label not in action['label_candidates']:
                action['label_candidates'].append(label)
        elif self.teacher_type == 'span':
            action['text'] = '{}\n{}'.format(passages, texts)
            action['labels'] = [span_label]
        return action

    def extract_passage_and_text(self, data, idx):
        passage_key = _first_key(data['dialog'][idx]['checked_sentence'])
        dialog_entry = data['dialog'][idx]
        text = passage = None
        if 'chosen' in passage_key:
            passage = ' '.join(data['chosen_topic_passage'])
            text = data['chosen_topic']
        elif 'self' in passage_key:
            passages = data['dialog'][idx - 2]['retrieved_passages']
            passage = None
            key = _first_val(dialog_entry['checked_passage'])
            for p in passages:
                if key in p:
                    passage = ' '.join(p[key])
                    break
            text = data['dialog'][idx - 2]['text']
        elif 'partner' in passage_key:
            passages = data['dialog'][idx - 1]['retrieved_passages']
            passage = None
            key = _first_val(dialog_entry['checked_passage'])
            for p in passages:
                if key in p:
                    passage = ' '.join(p[key])
                    break
            text = data['dialog'][idx - 1]['text']
        return (passage, text)

    def extract_passages_and_texts(self, d, idx):
        chosen_passages = ' '.join(d['chosen_topic_passage'])
        chosen_text = d['chosen_topic']
        if idx - 1 >= 0:
            appr_passages = d['dialog'][idx - 1]['retrieved_passages']
            appr_text = d['dialog'][idx - 1]['text']
            appr_list = []
            for passage in appr_passages:
                for v in passage.values():
                    temp = ' '.join(v)
                    appr_list.append(temp)
            appr = '\n'.join(appr_list)
        else:
            appr_passages = ''
            appr_text = ''
        if idx - 2 >= 0:
            wizard_passages = d['dialog'][idx - 2]['retrieved_passages']
            wizard_text = d['dialog'][idx - 2]['text']
            wizard_list = []
            for passage in wizard_passages:
                for v in passage.values():
                    temp = ' '.join(v)
                    wizard_list.append(temp)
            wizard = '\n'.join(wizard_list)
        else:
            wizard_passages = ''
            wizard_text = ''
        if idx - 2 >= 0:
            passages = '\n'.join([chosen_passages, wizard, appr])
            texts = ' '.join([chosen_text, wizard_text, appr_text])
        elif idx - 1 >= 0:
            passages = '\n'.join([chosen_passages, appr])
            texts = ' '.join([chosen_text, appr_text])
        else:
            passages = chosen_passages
            texts = chosen_text
        return (passages, texts)

    def get_span_label(self, data, idx):
        dialog_entry = data['dialog'][idx]
        said = dialog_entry['text']
        sentence = _first_val(dialog_entry['checked_sentence'])
        overlap = self.get_span(said, sentence)
        if not overlap or overlap in self.stop_words:
            label = sentence
        else:
            label = overlap
        return label