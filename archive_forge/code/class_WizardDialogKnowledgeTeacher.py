import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
class WizardDialogKnowledgeTeacher(WizardOfWikipediaTeacher):
    """
        Teacher that returns the following action dict:
        {
            'text': chosen_topic
 # if first ex in ep
                    last_apprentice_message
 # if possible
                    wizard_message # if --label-type is chosen_sent

            'knowledge': title_1 sentence_1

                                .
                                .
                                .
                         title_m sentence_n # all knowledge available to wizard
            'labels': [title_checked sentence_checked] # default
                                        OR
                      [wizard_response] # if --label-type set to 'response'

            'label_candidates': knowledge + [no_passages_used no_passages_used]
        }
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.label_type = opt.get('label_type', 'response')
        self.include_knowledge = opt.get('include_knowledge', True)
        self.include_checked_sentence = opt.get('include_checked_sentence', False)
        self.knowledge_separator = opt.get('include_knowledge_separator', False)
        self.chosen_topic_delimiter = opt.get('chosen_topic_delimiter', '\n')
        self.num_exs = sum((self.len_episode(i) for i in range(len(self.data))))

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Wizard Dialog Knowledge arguments')
        agent.add_argument('--label-type', type=str, choices=['response', 'chosen_sent'], default='response', help='whether to populate label field with the wizard response, or the chosen sentence')
        agent.add_argument('--include-knowledge', type='bool', default=True, help='Whether to include the knowledge available to the wizard')
        agent.add_argument('--include-checked-sentence', type='bool', default=True, help="Whether to include the Wizard'schecked sentence")
        agent.add_argument('--include-knowledge-separator', type='bool', default=False, help='include special __knowledge__ token between title and passage')
        agent.add_argument('--chosen-topic-delimiter', type=str, default='\n', help='delimiter used when including chosen topic')
        agent.add_argument('--num-topics', type=int, default=5, help='in interactive mode, this is the number of topic choicesthe human will have')

    def len_episode(self, ep):
        d = self.data[ep]
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        if wizard_first:
            return (len(d['dialog']) - 1) // 2
        return len(d['dialog']) // 2

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        episode_done = entry_idx == self.len_episode(episode_idx) - 1
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else entry_idx * 2 + 1
        apprentice_ret_passages = wizard_ret_passages = {}
        if not wizard_first or idx != 0:
            apprentice_entry = d['dialog'][idx - 1]
            apprentice_ret_passages = apprentice_entry['retrieved_passages']
        if idx - 2 >= 0:
            wizard_prev_entry = d['dialog'][idx - 2]
            wizard_ret_passages = wizard_prev_entry['retrieved_passages']
        chosen_topic = d.get('chosen_topic', '')
        chosen_topic_passages = d['chosen_topic_passage']
        chosen_topic = d.get('chosen_topic', '')
        knowledge_dict = {chosen_topic: chosen_topic_passages}
        for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
            for passage in ret_passes:
                for k, v in passage.items():
                    if k not in knowledge_dict.keys():
                        knowledge_dict[k] = v
        if idx == 0:
            text = chosen_topic
        elif idx == 1:
            text = f'{chosen_topic}{self.chosen_topic_delimiter}{apprentice_entry['text']}'
        else:
            text = ''
            if self.label_type == 'chosen_sent':
                text += '{}\n'.format(wizard_prev_entry['text'])
            text += apprentice_entry['text']
        wizard_entry = d['dialog'][idx]
        if self.label_type == 'response':
            labels = [wizard_entry['text']]
        else:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            if self.knowledge_separator and title != TOKEN_NOCHOSEN:
                labels = ['{} {} {}'.format(title, TOKEN_KNOWLEDGE, sentence)]
            else:
                labels = ['{} {}'.format(title, sentence)]
        label_cands = ['{} {}'.format(TOKEN_NOCHOSEN, TOKEN_NOCHOSEN)]
        knowledge_str = ''
        for title, passage in knowledge_dict.items():
            for p in passage:
                if self.knowledge_separator:
                    cand = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, p)
                else:
                    cand = '{} {}'.format(title, p)
                knowledge_str += cand + '\n'
                label_cands.append(cand)
        if self.label_type == 'response':
            if 'train' in self.datatype:
                label_cands = []
            else:
                label_cands = wizard_entry.get('candidate_responses', [])
        action = {'id': 'WizardDialogKnowledgeTeacher', 'text': text, 'labels': labels, 'chosen_topic': chosen_topic, 'episode_done': episode_done, 'label_candidates': label_cands}
        if self.include_knowledge:
            action['knowledge'] = knowledge_str
        if self.include_checked_sentence:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            action['title'] = title
            action['checked_sentence'] = sentence
        return action