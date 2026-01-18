from parlai.core.teachers import Teacher
from .build import build
import json
import os
import random
class TaskNTalkTeacher(Teacher):
    """
    TaskNTalk basic teacher, it picks a random image and associates a random task with
    it.

    Metric updates and observation are to be implemented.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'taskntalk'
        if not shared:
            self._setup_data(self.opt['datafile'])
        else:
            self.data = shared['data']
            self.task_defn = shared['task_defn']
            self.task_index = shared['task_index']

    def _setup_data(self, data_path):
        """
        Read the json file and store images and task definitions.
        """
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            json_data = json.load(data_file)
            self.data = json_data['data']
            self.task_defn = json_data['task_defn']
        self.task_index = {'color': 0, 'shape': 1, 'style': 2}
        random.shuffle(self.data)

    def share(self):
        """
        Share images and task definitions with other teachers.
        """
        shared = super().share()
        shared['data'] = self.data
        shared['task_defn'] = self.task_defn
        shared['task_index'] = self.task_index
        return shared

    def __len__(self):
        return len(self.data)

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        self.observation = observation
        return observation

    def act(self):
        """
        Select random image and associate random task with it.
        """
        image = random.choice(self.data)
        task = random.choice(self.task_defn)
        labels = [image[self.task_index[attr]] for attr in task]
        action = {'image': ' '.join(image), 'text': ' '.join(task), 'labels': [' '.join(labels)], 'episode_done': True}
        return action