import json
import os
import re
import numpy as np
from collections import Counter
from parlai.core.agents import Agent
from collections import defaultdict
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015
class VqaDictionaryAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        dictionary = argparser.add_argument_group('Dictionary Arguments')
        dictionary.add_argument('--dict-file', help='if set, the dictionary will automatically save to this path' + ' during shutdown')
        dictionary.add_argument('--dict-initpath', help='path to a saved dictionary to load tokens / counts from to ' + 'seed the dictionary with initial tokens and/or frequencies')
        dictionary.add_argument('--dict-maxexs', default=300000, type=int, help='max number of examples to build dict on')
        dictionary.add_argument('-smp', '--samplingans', type='bool', default=True)
        dictionary.add_argument('--nans', type=int, default=2000)
        dictionary.add_argument('--maxlength', type=int, default=16)
        dictionary.add_argument('--minwcount', type=int, default=0)
        dictionary.add_argument('--nlp', default='mcb')

    def __init__(self, opt, shared=None):
        super(VqaDictionaryAgent, self).__init__(opt)
        self.id = 'VqaDictionary'
        self.null_token = '__NULL__'
        self.unk_token = '__UNK__'
        if shared:
            self.freq = shared.get('freq', {})
            self.tok2ind = shared.get('tok2ind', {})
            self.ind2tok = shared.get('ind2tok', {})
            self.ans2ind = shared.get('ans2ind', {})
            self.ind2ans = shared.get('ind2ans', {})
        else:
            self.freq = defaultdict(int)
            self.ansfreq = defaultdict(int)
            self.ans2ques = defaultdict(list)
            self.tok2ind = {}
            self.ind2tok = {}
            self.ans2ind = {}
            self.ind2ans = {}
            if self.null_token:
                self.tok2ind[self.null_token] = 0
                self.ind2tok[0] = self.null_token
            if self.unk_token:
                index = len(self.tok2ind)
                self.tok2ind[self.unk_token] = index
                self.ind2tok[index] = self.unk_token
        if opt.get('dict_file') and os.path.isfile(opt['dict_file']):
            self.load(opt['dict_file'])
        if not shared:
            if self.null_token:
                self.freq[self.null_token] = 1000000002
            if self.unk_token:
                self.freq[self.unk_token] = 1000000000
            if opt.get('dict_file'):
                self.save_path = opt['dict_file']

    def __len__(self):
        return len(self.tok2ind)

    def add_to_ques_dict(self, tokens):
        """
        Builds dictionary from the list of provided tokens.

        Only adds words contained in self.embedding_words, if not None.
        """
        for token in tokens:
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def add_to_ans_dict(self, token):
        """
        Builds dictionary from the list of provided tokens.

        Only adds words contained in self.embedding_words, if not None.
        """
        self.ansfreq[token] += 1
        if token not in self.ans2ind:
            index = len(self.ans2ind)
            self.ans2ind[token] = index
            self.ind2ans[index] = token

    def tokenize_mcb(self, s):
        t_str = s.lower()
        for i in ['\\?', '\\!', "\\'", '\\"', '\\$', '\\:', '\\@', '\\(', '\\)', '\\,', '\\.', '\\;']:
            t_str = re.sub(i, '', t_str)
        for i in ['\\-', '\\/']:
            t_str = re.sub(i, ' ', t_str)
        q_list = re.sub('\\?', '', t_str.lower()).split(' ')
        q_list = list(filter(lambda x: len(x) > 0, q_list))
        return q_list

    def split_tokenize(self, s):
        return s.lower().replace('.', ' . ').replace('. . .', '...').replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ').replace('!', ' ! ').replace('?', ' ? ').split()

    def act(self):
        """
        Add any words passed in the 'text' field of the observation to this dictionary.
        """
        mc_label = self.observation.get('mc_label', self.observation.get('labels', []))
        for text in mc_label:
            self.ansfreq[text] += 1
            self.ans2ques[text].append(self.tokenize_mcb(self.observation.get('text')))
        return {'id': 'Dictionary'}

    def encode_question(self, examples, training):
        minwcount = self.opt.get('minwcount', 0)
        maxlength = self.opt.get('maxlength', 16)
        for ex in examples:
            words = self.tokenize_mcb(ex['text'])
            if training:
                words_unk = [w if self.freq.get(w, 0) > minwcount else self.unk_token for w in words]
            else:
                words_unk = [w if w in self.tok2ind else self.unk_token for w in words]
            ex['question_wids'] = [self.tok2ind[self.null_token]] * maxlength
            for k, w in enumerate(words_unk):
                if k < maxlength:
                    ex['question_wids'][k] = self.tok2ind[w]
        return examples

    def encode_answer(self, examples):
        for ex in examples:
            if self.opt.get('samplingans', True):
                labels = ex.get('labels', ex.get('eval_labels'))
                ans_count = Counter(labels).most_common()
                valid_ans = []
                valid_count = []
                for ans in ans_count:
                    if ans[0] in self.ans2ind:
                        valid_ans.append(self.ans2ind[ans[0]])
                        valid_count.append(ans[1])
                if not valid_ans:
                    ex['answer_aid'] = 0
                else:
                    probs = valid_count / np.sum(valid_count)
                    ex['answer_aid'] = int(np.random.choice(valid_ans, p=probs))
            else:
                ex['answer_aid'] = self.ans2ind[ex['mc_label'][0]]
        return examples

    def decode_answer(self, examples):
        txt_answers = []
        for ex in examples:
            txt_answers.append(self.ind2ans[ex])
        return txt_answers

    def load(self, filename):
        """
        Load pre-existing dictionary in 'token[<TAB>count]' format.

        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        print('Dictionary: loading dictionary from {}'.format(filename))
        with open(filename) as read:
            for line in read:
                split = line.strip().split('\t')
                token = unescape(split[0])
                cnt = int(split[1]) if len(split) > 1 else 0
                self.freq[token] = cnt
                if token not in self.tok2ind:
                    index = len(self.tok2ind)
                    self.tok2ind[token] = index
                    self.ind2tok[index] = token
        print('[ num ques words =  %d ]' % len(self.ind2tok))
        with open(filename[:-5] + '_ans.dict') as read:
            for line in read:
                split = line.strip().split('\t')
                token = unescape(split[0])
                cnt = int(split[1]) if len(split) > 1 else 0
                self.ansfreq[token] = cnt
                if token not in self.ans2ind:
                    index = len(self.ans2ind)
                    self.ans2ind[token] = index
                    self.ind2ans[index] = token
        print('[ num ans words =  %d ]' % len(self.ind2ans))

    def save(self, filename=None, append=False, sort=True):
        """
        Save dictionary to file. Format is 'token<TAB>count' for every token in the
        dictionary, sorted by count with the most frequent words first.

        If ``append`` (default ``False``) is set to ``True``, appends instead
        of overwriting.

        If ``sort`` (default ``True``), then first sort the dictionary before
        saving.
        """
        cw = sorted([(count, w) for w, count in self.ansfreq.items()], reverse=True)
        final_exs = cw[:self.opt.get('nans', 2000)]
        final_list = dict([(w, c) for c, w in final_exs])
        self.ansfreq = defaultdict(int)
        for ans, ques in self.ans2ques.items():
            if ans in final_list:
                for que in ques:
                    self.add_to_ques_dict(que)
                self.add_to_ans_dict(ans)
        filename = self.opt['dict_file'] if filename is None else filename
        print('Dictionary: saving dictionary to {}'.format(filename))
        with open(filename, 'a' if append else 'w') as write:
            for i in range(len(self.ind2tok)):
                tok = self.ind2tok[i]
                cnt = self.freq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))
        with open(filename[:-5] + '_ans.dict', 'a' if append else 'w') as write:
            for i in range(len(self.ind2ans)):
                tok = self.ind2ans[i]
                cnt = self.ansfreq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))

    def shutdown(self):
        """
        Save on shutdown if ``save_path`` is set.
        """
        if hasattr(self, 'save_path'):
            self.save(self.save_path)