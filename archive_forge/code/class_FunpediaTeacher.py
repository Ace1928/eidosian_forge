from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
class FunpediaTeacher(FixedDialogTeacher):
    """
    Generic teacher which extracts each of the fields.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared:
            self.entries = shared['entries']
        else:
            self.entries = []
            self.datafile = _path(opt)
            self._setup_data()
        self.reset()

    def share(self):
        shared = super().share()
        shared['entries'] = self.entries
        return shared

    def num_episodes(self):
        return self.num_examples()

    def num_examples(self):
        return len(self.entries)

    def _setup_data(self):
        title_prefix = '1 passage title: '
        title_prefix_len = len(title_prefix)
        persona_prefix = '2 personality: '
        persona_prefix_len = len(persona_prefix)
        text_prefix = '3 '
        text_prefix_len = len(text_prefix)
        data_reader = grouper(_strip_reader(self.datafile), 3, '')
        for title, persona, text in data_reader:
            if not title:
                break
            assert title.startswith(title_prefix)
            title = title[title_prefix_len:]
            assert persona.startswith(persona_prefix)
            persona = persona[persona_prefix_len:]
            assert text.startswith(text_prefix)
            text = text[text_prefix_len:]
            passage, label = text.split('\t')
            self.entries.append({'title': title, 'label': label, 'passage': passage, 'persona': persona})

    def get_text(self, entry):
        return '\n'.join([entry['title'], entry['persona'], entry['passage']])

    def _build_action(self, entry):
        return {'text': self.get_text(entry), 'labels': [entry['label']], 'reward': 0, 'episode_done': True}

    def get(self, episode_idx, entry_idx=0):
        assert entry_idx == 0
        return self._build_action(self.entries[episode_idx])