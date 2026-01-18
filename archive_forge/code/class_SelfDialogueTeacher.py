from parlai.core.teachers import FixedDialogTeacher
from . import tm_utils
import json
class SelfDialogueTeacher(FixedDialogTeacher):
    """
    Teacher for written two-person dialogues with labels being responses for the
    previous statement.

    The data is traversed twice (doubled), once for modelling USER replies and once for
    modelling ASSISTANT replies.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        opt['fn'] = 'self-dialogs.json'
        if shared and 'convos' in shared:
            self.convos = shared['convos']
            self.ep_cheat_sheet = shared['ep_cheat_sheet']
            self.num_ex = shared['num_ex']
        else:
            self.ep_cheat_sheet = {}
            data_path = tm_utils._path(opt)
            self.num_ex = 0
            self._setup_data(data_path, opt)
        self.reset()

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)
        convos_update = []
        for convo in self.convos:
            conversation = convo['utterances']
            if len(conversation) > 1:
                self.ep_cheat_sheet[len(self.ep_cheat_sheet)] = tm_utils.gen_ep_cheatsheet(conversation)
                curr_cheatsheet = self.ep_cheat_sheet[len(self.ep_cheat_sheet) - 1]
                self.num_ex += curr_cheatsheet[tm_utils.USER_NUM_EX] + curr_cheatsheet[tm_utils.ASSIS_NUM_EX]
                convos_update += [conversation]
        self.convos = convos_update

    def num_examples(self):
        return self.num_ex

    def num_episodes(self):
        return len(self.convos) * 2

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx % len(self.convos)]
        if episode_idx < len(self.convos):
            ep_done = entry_idx * 2 == self.ep_cheat_sheet[episode_idx][tm_utils.LAST_USER_IDX]
            predecessor = conversation[entry_idx * 2]['text']
            successor = conversation[entry_idx * 2 + 1]['text']
        else:
            ep_done = entry_idx * 2 + 1 == self.ep_cheat_sheet[episode_idx % len(self.convos)][tm_utils.LAST_ASSISTANT_IDX]
            predecessor = conversation[entry_idx * 2 + 1]['text']
            successor = conversation[entry_idx * 2 + 2]['text']
        action = {'id': self.id, 'text': predecessor, 'episode_done': ep_done, 'labels': [successor]}
        return action