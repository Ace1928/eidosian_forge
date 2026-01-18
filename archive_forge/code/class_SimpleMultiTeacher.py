from parlai.core.teachers import ParlAIDialogTeacher
from .build import build
import copy
import os
import random
class SimpleMultiTeacher(DefaultTeacher):

    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('LIGHT Dialogue options')
        agent.add_argument('--light_use_repeat', type=str, default='none', choices=['self_last', 'partner_last', 'none', 'both_last'])
        agent.add_argument('--light_use_setting', type='bool', default=True)
        agent.add_argument('--light_unseen_test', type='bool', default=False)
        agent.add_argument('--light_use_person_names', type='bool', default=True)
        agent.add_argument('--light_use_persona', type=str, default='self', choices=['partner', 'self', 'all', 'none'])
        agent.add_argument('--light_use_taskname', type='bool', default=False)
        agent.add_argument('--light_use_objects', type='bool', default=False)
        agent.add_argument('--light_use_emote', type=str, default='none', choices=['partner', 'self', 'all', 'none'])
        agent.add_argument('--light_use_speech', type=str, default='partner', choices=['partner', 'self', 'all', 'none'])
        agent.add_argument('--light_use_action', type=str, default='none', choices=['partner', 'self', 'all', 'none'])
        agent.add_argument('--light_use_affordances', type='bool', default=False)
        agent.add_argument('--light_use_current_self_output', type=str, default='all', choices=['none', 'all', 'all_filtered', 'all_filtered_remove'])
        agent.add_argument('--light_label_type', type=str, default='speech', choices=['speech', 'action', 'emote', 'which'], help='type of target in light dialogues')
        agent.add_argument('--light_use_cands', type=int, default=20)
        agent.add_argument('--light_use_clip_cands', type=int, default=10000)
        agent.add_argument('--light_use_speech_prefix', type='bool', default=False)
        agent.add_argument('--light_percent_train_exs', type=float, default=1.0)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id += '_' + self.opt['light_label_type']