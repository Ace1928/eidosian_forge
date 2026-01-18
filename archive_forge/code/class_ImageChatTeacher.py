import json
import os
from typing import Tuple
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.utils.typing import TShared
from .build import build
class ImageChatTeacher(FixedDialogTeacher):
    """
    Provides the personality in the `text` field, and response in the `labels` field.

    To specify your own path to the YFCC100m images, please use the `--yfcc-path`
    command line argument.
    """

    def __init__(self, opt: Opt, shared: TShared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.image_mode = opt.get('image_mode', 'no_image_model')
        self.data_path, personalities_data_path, self.image_path = _path(opt)
        self.datatype = opt['datatype'].split(':')[0]
        self.include_personality = opt.get('include_personality')
        self.include_image = opt.get('include_image') and opt.get('load_images')
        self.num_cands = opt.get('num_cands')
        if shared and 'data' in shared:
            self.data = shared['data']
            self.personalities = shared['personalities']
            self.image_loader = shared['image_loader']
        else:
            self.image_loader = ImageLoader(opt)
            self._setup_data(self.data_path, personalities_data_path)
        self.num_exs = sum((len(d['dialog']) for d in self.data))
        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Personality-Captions arguments')
        agent.add_argument('--include-personality', type='bool', default=True, help='Whether to provide personality to agent')
        agent.add_argument('--include-image', type='bool', default=True, help='Whether to provide image to agent')
        agent.add_argument('--yfcc-path', type=str, default=None, help='Path to yfcc images (if not downloaded via the provided download script)')
        agent.add_argument('--load-images', type='bool', default=True, help='Specify whether to load images')
        agent.add_argument('--num-cands', type=str, default='100', choices=['100', '1000'], help='how many candidates to provide agent')

    def _setup_data(self, data_path: str, personalities_data_path: str):
        """
        Load the data.
        """
        print('loading: ' + data_path)
        with open(data_path) as f:
            self.data = json.load(f)
        with open(personalities_data_path) as f:
            self.personalities = json.load(f)

    def reset(self):
        """
        Override to Reset self.example.
        """
        super().reset()
        self.example = None

    def num_episodes(self) -> int:
        return len(self.data)

    def num_examples(self) -> int:
        return self.num_exs

    def submit_load_request(self, image_id: str):
        img_path = os.path.join(self.image_path, '{}.jpg'.format(image_id))
        self.data_loader.request_load(self.receive_data, self.image_loader.load, (img_path,))

    def get(self, episode_idx: int, entry_idx: int=0):
        data = self.data[episode_idx]
        personality, text = data['dialog'][entry_idx]
        episode_done = entry_idx == len(data['dialog']) - 1
        action = {'text': personality if self.include_personality else '', 'image_id': data['image_hash'], 'episode_done': episode_done, 'labels': [text]}
        if 'candidates' in data:
            action['label_candidates'] = data['candidates'][entry_idx][self.num_cands]
        return action

    def next_example(self) -> Tuple[Message, bool]:
        """
        Returns the next example from this dataset after starting to queue up the next
        example.

        :return (example, epoch done):
            returns the next example as well as whether the epoch is done.
        """
        ready = None
        load_image = self.image_mode != 'no_image_model' and self.include_image
        if self.example is not None:
            if load_image and 'image_id' in self.example:
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.imageEpochDone)
        self.example, self.imageEpochDone = super().next_example()
        if load_image and 'image_id' in self.example:
            image_id = self.example['image_id']
            self.submit_load_request(image_id)
        if ready is None:
            return self.next_example()
        else:
            return ready

    def share(self) -> TShared:
        shared = super().share()
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        shared['personalities'] = self.personalities
        return shared