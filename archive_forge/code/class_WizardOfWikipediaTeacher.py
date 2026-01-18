import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
class WizardOfWikipediaTeacher(FixedDialogTeacher):
    """
    The default teacher; essentially reads the json file and outputs the raw data.

    Actions have the following form:
    {
        'wizard_eval': <evaluation of wizard>,
        'chosen_topic': <chosen_topic>,
        'chosen_topic_passage': <chosen topic passage>,
        'mtdo': <whether the conversation had sufficient overlap>,
        'text': <text>
        'retrieved_topics': <topics retrieved for text>
        'full_retrieved_passages': <full retrieved passages>
        'retrieved_passages': <passages shown to turker>
        'checked_sentence': <checked sentence if wizard, else None>
        'checked_passage': <checked_passage if wizard, else None>
    }

    The 'passages' are lists of 1 entry dicts, mapping a topic to the sentences

    Specify the valid/test split after the last colon in the task, e.g.
    wizard_of_wikipedia:<teacher>:random_split
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        task = opt.get('task', 'wizard_of_wikipedia:WizardOfWikipedia:random_split')
        split = task.split(':')
        split = split[2] if len(split) == 3 else 'random_split'
        opt['task'] = 'wizard_of_wikipedia'
        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.data_path = _path(opt, split=split)
            self._setup_data()
        self.num_exs = sum((len(d['dialog']) for d in self.data))
        self.reset()

    def _setup_data(self):
        print('loading: ' + self.data_path)
        with open(self.data_path) as f:
            self.data = json.load(f)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        dialog_entry = d['dialog'][entry_idx]
        episode_done = entry_idx == len(d['dialog']) - 1
        action = {'wizard_eval': d['wizard_eval'], 'chosen_topic': d['chosen_topic'], 'chosen_topic_passage': d['chosen_topic_passage'], 'text': dialog_entry['text'], 'retrieved_topics': dialog_entry['retrieved_topics'], 'retrieved_passages': dialog_entry['retrieved_passages'], 'checked_sentence': dialog_entry.get('checked_sentence', None), 'checked_passage': dialog_entry.get('checked_passage', None), 'episode_done': episode_done}
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared