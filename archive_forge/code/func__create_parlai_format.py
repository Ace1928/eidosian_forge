import json
import os
from parlai.core import build_data
def _create_parlai_format(dpath: str):
    """
    Copy data into the format read by ParlAIDialogTeacher.

    'text' will be from the free Turker, who speaks first, and 'label' will be from the
    guided Turker.
    """
    datatypes = ['train', 'valid', 'test']
    for datatype in datatypes:
        load_path = os.path.join(dpath, f'{datatype}.json')
        save_path = os.path.join(dpath, f'{datatype}.txt')
        print(f'Loading {load_path}.')
        with open(load_path, 'r', encoding='utf8') as f_read:
            data = json.load(f_read)
        print(f'Saving to {save_path}')
        with open(save_path, 'w', encoding='utf8') as f_write:
            for episode in data:
                assert len(episode['dialog']) == len(episode['suggestions']) == len(episode['chosen_suggestions'])
                num_entries = len(episode['dialog']) // 2
                for entry_idx in range(num_entries):
                    line = _get_line(episode=episode, num_entries=num_entries, entry_idx=entry_idx)
                    f_write.write(f'{line} \n')