import os
import json
from parlai.core.teachers import FixedDialogTeacher
from .build import build
class NoStartTeacher(Convai2Teacher):
    """
    Same as default teacher, but it doesn't contain __SILENCE__ entries.

    If we are the first speaker, then the first utterance is skipped.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.num_exs = sum((len(d['dialogue']) - 1 for d in self.data))
        self.all_eps = self.data + [d for d in self.data if len(d['dialogue']) > 2]
        self.num_eps = len(self.all_eps)

    def get(self, episode_idx, entry_idx=0):
        full_eps = self.all_eps[episode_idx]
        entries = full_eps['dialogue']
        speaker_id = int(episode_idx >= len(self.data))
        their_turn = entries[speaker_id + 2 * entry_idx]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]
        episode_done = 2 * entry_idx + speaker_id + 1 >= len(entries) - 2
        action = {'topic': full_eps['topic'], 'text': their_turn['text'], 'emotion': their_turn['emotion'], 'act_type': their_turn['act'], 'labels': [my_turn['text']], 'episode_done': episode_done}
        return action