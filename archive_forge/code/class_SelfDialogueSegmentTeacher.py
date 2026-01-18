from parlai.core.teachers import FixedDialogTeacher
from . import tm_utils
import json
class SelfDialogueSegmentTeacher(FixedDialogTeacher):
    """
    Teacher for written two-person dialogues with labels being relevant/useful parts in
    the input sentence.

    The different datatypes of the labels within the data have also been encoded as
    `label_types`
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        opt['fn'] = 'self-dialogs.json'
        if shared and 'convos' in shared:
            self.convos = shared['convos']
            self.num_ex = shared['num_ex']
        else:
            data_path = tm_utils._path(opt)
            self.num_ex = 0
            self._setup_data(data_path, opt)
        self.reset()

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx]
        conv_len = len(conversation['utterances'])
        utterance = conversation['utterances'][entry_idx]['text']
        ep_done = entry_idx == conv_len - 1
        action = {'id': self.id, 'text': utterance, 'episode_done': ep_done}
        action['labels'] = []
        action['label_types'] = []
        segments = conversation['utterances'][entry_idx]['segments']
        for segment in segments:
            action['labels'] += [segment['text']]
            tmp = []
            for annot in segment['annotations']:
                tmp += [annot['name']]
            action['label_types'] += [tmp]
        return action

    def num_examples(self):
        return self.num_ex

    def num_episodes(self):
        return len(self.convos)

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)
        convos_updated = []
        for convo in self.convos:
            updated_dialog = []
            for i in range(len(convo['utterances'])):
                if 'segments' in convo['utterances'][i]:
                    updated_dialog += [convo['utterances'][i]]
            convo['utterances'] = updated_dialog
            if convo['utterances']:
                convos_updated += [convo]
                self.num_ex += len(convo['utterances'])
        self.convos = convos_updated