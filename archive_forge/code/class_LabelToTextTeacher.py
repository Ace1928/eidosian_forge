import copy
from abc import ABC, abstractmethod
from parlai.core.agents import create_agent_from_shared
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname, Teacher
class LabelToTextTeacher(AbstractWrapperTeacher):
    """
    Teacher that will shift message['labels'][0] into message['text'] for whatever task
    is specified with --wrapper-task.

    Because the dialogue history is effectively overwritten by this action, all episodes
    will be flattened into one example each.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)

    def act(self):
        """
        Act on the previous observation.
        """
        act = self.task.act()
        new_act = copy.deepcopy(act)
        if 'labels' in act or 'eval_labels' in act:
            labels_type = 'labels' if 'labels' in act else 'eval_labels'
            labels = act[labels_type]
            if len(labels) != 1:
                raise ValueError('LabelToTextTeacher can only be used with one label!')
            new_act.force_set('text', labels[0])
            new_act.force_set(labels_type, [''])
        else:
            assert 'text' not in act and act['episode_done'] is True
        new_act.force_set('episode_done', True)
        return new_act