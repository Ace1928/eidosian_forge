import copy
import json
import os
import re
import numpy as np
from parlai.core.teachers import ParlAIDialogTeacher, MultiTaskTeacher
from projects.self_feeding.utils import add_person_tokens
from .build import build
class SelfFeedingTeacher(ParlAIDialogTeacher):
    """
    Teacher for the SelfFeedingAgent.

    opt['datatype'] determines whether we use the designated filepath ('train') or one
    of the eval files ('valid', 'test'), which are identical regardless of     what
    training set is being used.
    """

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if 'subtask' not in opt:
            print('Warning: SelfFeedingteacher should be assigned subtask. Defaulting to dialog')
            opt['subtask'] = 'dialog'
        if 'train' in opt['datatype']:
            train_file_flag = f'{opt['subtask'][:3]}_train'
            if opt.get(train_file_flag, None):
                path = _path(opt, opt[train_file_flag], add_suffix=False)
            else:
                path = _path(opt, 'train', add_suffix=True)
        else:
            eval_file_flag = f'{opt['subtask'][:3]}_{opt['datatype']}'
            if opt.get(eval_file_flag, None):
                path = _path(opt, opt[eval_file_flag], add_suffix=False)
            else:
                path = _path(opt, opt['datatype'].split(':')[0], add_suffix=True)
        if not os.path.exists(path):
            raise ValueError('Unrecognized filepath: {}'.format(path))
        opt['parlaidialogteacher_datafile'] = path
        opt['datafile'] = path
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        project = argparser.add_argument_group('Self-Feeding Tasks')
        project.add_argument('-st', '--subtasks', type=str, help='comma-separated list of tasks used by MTL teacher')
        project.add_argument('-dia-train', '--dia-train', type=str, help='the filename to train on for the dialog task')
        project.add_argument('-fee-train', '--fee-train', type=str, help='the filename to train on for the feedback task')
        project.add_argument('-sat-train', '--sat-train', type=str, help='the filename to train on for the satisfaction task')
        project.add_argument('-dia-valid', '--dia-valid', type=str, help='the filename to eval on for the dialog task')
        project.add_argument('-fee-valid', '--fee-valid', type=str, help='the filename to eval on for the feedback task')
        project.add_argument('-sat-valid', '--sat-valid', type=str, help='the filename to eval on for the satisfaction task')
        project.add_argument('-dia-test', '--dia-test', type=str, help='the filename to eval on for the dialog task')
        project.add_argument('-fee-test', '--fee-test', type=str, help='the filename to eval on for the feedback task')
        project.add_argument('-sat-test', '--sat-test', type=str, help='the filename to eval on for the satisfaction task')
        project.add_argument('-trial', '--trial', type=int, default=0, help='the index of a repeated trial (not used in code)')
        project.add_argument('-mt', '--max-train', type=int, default=0, help='if non-zero, only the first max-train examples will be used if it is read by an instance of ParlaiDialogTeacher')
        argparser.set_defaults(history_size=2)

    def _setup_data(self, path):
        """
        Reads data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.
        """
        print('[ Loading Self-Feeding text data:' + path + ']')
        self.episodes = []
        self.num_exs = 0
        self.max_train = self.opt.get('max_train', 0)
        with open(path, 'r') as f:
            for line in f.readlines():
                if self.max_train and self.num_exs >= self.max_train:
                    break
                parley = json.loads(line)
                if self.opt['history_size'] == 0:
                    parley['context'] = '__null__'
                elif self.opt['history_size'] > 0:
                    utterances = re.split('__p\\d__', parley['context'])[1:]
                    trimmed = utterances[-self.opt['history_size']:]
                    parley['context'] = add_person_tokens(trimmed, last_speaker=1)
                parley['memories'] = []
                episode = {'text': parley['context'], 'labels': [parley['response']], 'label_candidates': parley.get('candidates', []), 'reward': parley.get('reward', 0), 'episode_done': True}
                episode['labels'] = [str(l) for l in episode['labels']]
                self.num_exs += 1
                self.episodes.append([episode])