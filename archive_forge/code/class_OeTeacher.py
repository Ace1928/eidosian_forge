import json
import os
import re
import numpy as np
from collections import Counter
from parlai.core.agents import Agent
from collections import defaultdict
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015
class OeTeacher(FixedDialogTeacher):
    """
    VQA Open-Ended teacher, which loads the json vqa data and implements its own `act`
    method for interacting with student agent.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        data_path, annotation_path, self.image_path = _path(opt)
        self.datafile = data_path
        self.image_mode = opt.get('image_mode', 'no_image_model')
        if shared and 'ques' in shared:
            self.ques = shared['ques']
            if 'annotation' in shared:
                self.annotation = shared['annotation']
            self.image_loader = shared['image_loader']
        else:
            self._setup_data(data_path, annotation_path)
            self.image_loader = ImageLoader(opt)
        self.reset()

    def reset(self):
        super().reset()
        self.example = None
        self.imageEpochDone = False

    def num_examples(self):
        """
        Number of examples in VQA-v1.
        """
        return len(self.ques['questions'])

    def num_episodes(self):
        return self.num_examples()

    def submit_load_request(self, image_id):
        img_path = self.image_path + '%012d.jpg' % image_id
        self.data_loader.request_load(self.receive_data, self.image_loader.load, (img_path,))

    def get(self, episode_idx, entry_idx=0):
        qa = self.ques['questions'][episode_idx]
        question = qa['question']
        action = {'text': question, 'image_id': qa['image_id'], 'episode_done': True}
        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [ans['answer'] for ans in anno['answers']]
        return action

    def next_example(self):
        """
        Returns the next example from this dataset after starting to queue up the next
        example.
        """
        ready = None
        if self.example is not None:
            if self.image_mode != 'no_image_model' and 'image_id' in self.example:
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.imageEpochDone)
        self.example, self.imageEpochDone = super().next_example()
        if self.image_mode != 'no_image_model' and 'image_id' in self.example:
            image_id = self.example['image_id']
            self.submit_load_request(image_id)
        if ready is None:
            return self.next_example()
        else:
            return ready

    def share(self):
        shared = super().share()
        shared['ques'] = self.ques
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        shared['image_loader'] = self.image_loader
        return shared

    def _setup_data(self, data_path, annotation_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)
        if not self.datatype.startswith('test'):
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)