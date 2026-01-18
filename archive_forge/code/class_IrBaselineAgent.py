import math
from collections.abc import Sequence
import heapq
import json
import torch
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
class IrBaselineAgent(Agent):
    """
    Information Retrieval baseline.
    """

    @staticmethod
    def add_cmdline_args(parser):
        """
        Add command line args specific to this agent.
        """
        parser = parser.add_argument_group('IrBaseline Arguments')
        parser.add_argument('-lp', '--length_penalty', type=float, default=0.5, help='length penalty for responses')
        parser.add_argument('-hsz', '--history_size', type=int, default=1, help='number of utterances from the dialogue history to take use as the query')
        parser.add_argument('--label_candidates_file', type=str, default=None, help='file of candidate responses to choose from')

    def __init__(self, opt, shared=None):
        """
        Initialize agent.
        """
        super().__init__(opt)
        self.id = 'IRBaselineAgent'
        self.length_penalty = float(opt['length_penalty'])
        self.dictionary = DictionaryAgent(opt)
        self.opt = opt
        self.history = []
        self.episodeDone = True
        if opt.get('label_candidates_file'):
            f = open(opt.get('label_candidates_file'))
            self.label_candidates = f.read().split('\n')

    def reset(self):
        """
        Reset agent properties.
        """
        self.observation = None
        self.history = []
        self.episodeDone = True

    def observe(self, obs):
        """
        Store and remember incoming observation message dict.
        """
        self.observation = obs
        self.dictionary.observe(obs)
        if self.episodeDone:
            self.history = []
        if 'text' in obs:
            self.history.append(obs.get('text', ''))
        self.episodeDone = obs.get('episode_done', False)
        return obs

    def act(self):
        """
        Generate a response to the previously seen observation(s).
        """
        if self.opt.get('datatype', '').startswith('train'):
            self.dictionary.act()
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()
        cands = None
        if obs.get('label_candidates', False) and len(obs['label_candidates']) > 0:
            cands = obs['label_candidates']
        if hasattr(self, 'label_candidates'):
            cands = self.label_candidates
        if cands:
            hist_sz = self.opt.get('history_size', 1)
            left_idx = max(0, len(self.history) - hist_sz)
            text = ' '.join(self.history[left_idx:len(self.history)])
            rep = self.build_query_representation(text)
            reply['text_candidates'] = rank_candidates(rep, cands, self.length_penalty, self.dictionary)
            reply['text'] = reply['text_candidates'][0]
        else:
            reply['text'] = "I don't know."
        return reply

    def save(self, path=None):
        """
        Save dictionary tokenizer if available.
        """
        path = self.opt.get('model_file', None) if path is None else path
        if path:
            self.dictionary.save(path + '.dict')
            data = {}
            data['opt'] = self.opt
            with open(path, 'wb') as handle:
                torch.save(data, handle)
            with open(path + '.opt', 'w') as handle:
                json.dump(self.opt, handle)

    def load(self, fname):
        """
        Load internal dictionary.
        """
        self.dictionary.load(fname + '.dict')

    def build_query_representation(self, query):
        """
        Build representation of query, e.g. words or n-grams.

        :param query: string to represent.

        :returns: dictionary containing 'words' dictionary (token => frequency)
                  and 'norm' float (square root of the number of tokens)
        """
        rep = {}
        rep['words'] = {}
        words = [w for w in self.dictionary.tokenize(query.lower())]
        rw = rep['words']
        used = {}
        for w in words:
            if len(self.dictionary.freq) > 0:
                rw[w] = 1.0 / (1.0 + math.log(1.0 + self.dictionary.freq[w]))
            elif w not in stopwords:
                rw[w] = 1
            used[w] = True
        rep['norm'] = math.sqrt(len(words))
        return rep