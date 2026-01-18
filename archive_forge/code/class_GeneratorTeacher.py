import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
class GeneratorTeacher(WizardDialogKnowledgeTeacher):
    """
    Teacher for training a generator.

    Depending on certain flag configurations, the teacher will include differing amounts
    of knowledge
    """

    def __init__(self, opt, shared=None):
        opt['label_type'] = 'response'
        opt['include_checked_sentence'] = True
        super().__init__(opt, shared)
        self.knowledge_separator = opt.get('include_knowledge_separator', True)
        self.only_checked_knowledge = opt.get('only_checked_knowledge', False)
        self.prepend_gold_knowledge = opt.get('prepend_gold_knowledge')
        self.gold_knowledge_delimiter = opt.get('gold_knowledge_delimiter', '\n')
        self.dropout = opt.get('ignorant_dropout', 0.0)

    @staticmethod
    def add_cmdline_args(argparser):
        argparser.set_defaults(include_knowledge_separator=True)
        WizardDialogKnowledgeTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('GeneratorTeacher Arguments')
        agent.add_argument('--only-checked-knowledge', type='bool', default=False, help='If true, only the checked sentence is provided')
        agent.add_argument('--ignorant-dropout', type=float, default=0.0, help='Eliminate all knowledge with this probability.Specify 1 for completely ignorant teacher')
        agent.add_argument('--prepend-gold-knowledge', type='bool', default=False, help='If true, prepend text with checked sentence')
        agent.add_argument('--gold-knowledge-delimiter', type=str, default='\n', help='delimiter for prepending gold knowledge')

    def getID(self):
        return 'WizTeacher'

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        if 'knowledge' not in a:
            return a
        a['label_candidates'] = []
        if not a['knowledge'].startswith(TOKEN_NOCHOSEN):
            a['knowledge'] = TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN + '\n' + a['knowledge']
        if self.only_checked_knowledge:
            a['knowledge'] = a['title'] + ' ' + TOKEN_KNOWLEDGE + ' ' + a['checked_sentence']
        if random.random() < self.dropout:
            a['title'] = TOKEN_NOCHOSEN
            a['checked_sentence'] = TOKEN_NOCHOSEN
            a['knowledge'] = TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN
        elif self.prepend_gold_knowledge:
            a['text'] = f'{TOKEN_KNOWLEDGE} {a['checked_sentence']} {TOKEN_END_KNOWLEDGE}{self.gold_knowledge_delimiter}{a['text']}'
        return a