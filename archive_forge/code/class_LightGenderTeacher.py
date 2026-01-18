from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.light_dialog.agents import DefaultTeacher as OrigLightTeacher
from parlai.tasks.light_genderation_bias.build import build
from collections import deque
from copy import deepcopy
import csv
import json
import os
import random
class LightGenderTeacher(FixedDialogTeacher):
    """
    ALL LIGHT Teacher: combines all gender bias mitigation methods described in.

    <https://arxiv.org/abs/1911.03842>.

    This teacher combines methods for gender bias mitigation in
    dialogue generation on the LIGHT dialogue task, including:
    - Adding more positive data (Pos Data):`--add-new-data`
    - Conditional training (Bias Ctrl): `--add-conditional`
    - Counterfactual Data Augmentation (CDA): `--add-counterfactual`

    For more information, please see our projects page:
    <https://parl.ai/projects/genderation_bias/>
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('LIGHT Gender Args')
        OrigLightTeacher.add_cmdline_args(parser)
        parser.add_argument('--add-new-data', type='bool', default=True, help='Add the extra data collected')
        parser.add_argument('--add-conditional', type='bool', default=True, help='Append the conditional bucket to the end of the input or not')
        parser.add_argument('--add-counterfactual', type='bool', default=True, help='Load counterfactual data for training')
        parser.add_argument('--force-conditional', type=str, default='none', help='Add one bucket to everything, regardless of the true bin')
        parser.add_argument('--bucket-only', type=str, default='none', help='Only train/evaluate on examples from a specific bucket')
        parser.set_defaults(light_use_objects=False, light_use_setting=False, light_use_affordances=False, light_use_persona_names=False, light_use_persona='none', light_use_action='none', light_use_emote='none', use_reply='none')
        return parser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.gender_dict = read_gender_tsv(os.path.join(_path(opt), GENDERED_LIST))
        self.fixed_random = random.Random(42)
        self.add_conditional = opt['add_conditional']
        self.add_counterfactual = opt['add_counterfactual']
        self.add_new_data = opt['add_new_data']
        self.force_conditional = opt['force_conditional']
        self.bucket_only = opt['bucket_only']
        if self.force_conditional == 'none':
            self.force_conditional = None
        if self.bucket_only == 'none':
            self.bucket_only = None
        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.data = self._setup_data(opt)
        super().__init__(opt, shared)
        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def get_bucket(self, text):
        """
        Get the bin or "bucket" that a string falls into.

        The bucket is determined based on the presence of male or femael gendered words
        in the text.
        """
        text = format_text(text)
        f_count, m_count, n_count = get_finegrained_count(text, self.gender_dict)
        if f_count == 0:
            f_bucket = 'f0'
        else:
            f_bucket = 'f1'
        if m_count == 0:
            m_bucket = 'm0'
        else:
            m_bucket = 'm1'
        all_bucket = ' '.join([f_bucket, m_bucket])
        return all_bucket

    def _flip_str(self, txt_lines):
        new_lines = []
        lines = txt_lines.split('\n')
        for text in lines:
            f_text = format_text(text)
            f_text_lst = f_text.split(' ')
            new_words = []
            for word in f_text_lst:
                if word in self.swap_dct:
                    if word == 'her':
                        random_choice = random.choice([0, 1])
                        if random_choice:
                            new_word = 'his'
                        else:
                            new_word = 'him'
                    else:
                        new_word = self.swap_dct[word]['word']
                else:
                    new_word = word
                new_words.append(new_word)
            new_f_text = ' '.join(new_words)
            uf_text = unformat_text(new_f_text)
            new_lines.append(uf_text)
        return '\n'.join(new_lines)

    def _flip_ex(self, ex):
        """
        Return the counterfactual example for a given example (i.e. swap 'he' --> 'she')
        """
        new_ex = deepcopy(ex)
        text = ex['text']
        labels = 'labels' if 'labels' in ex else 'eval_labels'
        label = ex[labels][0]
        new_ex.force_set('text', self._flip_str(text))
        new_ex.force_set(labels, [self._flip_str(label)])
        return new_ex

    def _get_new_data(self, opt):
        """
        Load extra positive dialogue data IFF datatype==train.
        """
        dt = opt['datatype'].split(':')[0]
        if dt == 'train':
            with open(os.path.join(_path(opt), NEW_DATA), 'r') as f:
                data = json.load(f)
            new_data = []
            for ep in data:
                new_ep = []
                for ex in ep:
                    ex['new_data'] = True
                    new_ep.append(Message(ex))
                new_data.append(new_ep)
            return new_data
        return []

    def _setup_data(self, opt):
        """
        Load original LIGHT dataset.
        """
        dt = opt['datatype'].split(':')[0]
        orig_episodes = OrigLightTeacher(opt).episodes
        if self.add_new_data:
            new_data = self._get_new_data(opt)
            total_data = orig_episodes + new_data
            self.fixed_random.shuffle(total_data)
            orig_episodes = total_data
        flat_episodes = []
        for ep in orig_episodes:
            flattened_ep = flatten(ep, -1, include_labels=True, delimiter='\n')
            flat_episodes += flattened_ep
        if self.add_counterfactual and dt != 'test':
            with open(os.path.join(_path(opt), COUNTERFACTUALS), 'rb') as f:
                self.swap_dct = json.load(f)
            new_episodes = []
            for ex in flat_episodes:
                new_ex = self._flip_ex(ex)
                ex['counterfactual'] = False
                new_ex['counterfactual'] = True
                new_episodes.append(ex)
                new_episodes.append(new_ex)
            flat_episodes = new_episodes
        bucket_percentages = {}
        new_episodes = []
        for ex in flat_episodes:
            label_type = 'labels' if 'labels' in ex else 'eval_labels'
            label = ex[label_type][0]
            bucket_key = self.get_bucket(label)
            bucket_percentages.setdefault(bucket_key, 0)
            bucket_percentages[bucket_key] += 1
            if self.add_conditional:
                if self.force_conditional is None:
                    new_text = ex['text'] + '\n' + bucket_key
                else:
                    new_text = ex['text'] + self.force_conditional
                ex.force_set('text', new_text)
            ex['bucket'] = bucket_key
            if self.bucket_only is None or self.bucket_only == bucket_key:
                new_episodes.append(ex)
        print('Distribution of bins:')
        total = sum(bucket_percentages.values())
        strs = []
        for k, v in bucket_percentages.items():
            pct = round(v / total * 100, 2)
            strs.append(f'{k}: {pct}%')
        strs = sorted(strs)
        for string in strs:
            print(string)
        return new_episodes

    def get(self, episode_idx, entry_idx=0):
        return Message(self.data[episode_idx])

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared