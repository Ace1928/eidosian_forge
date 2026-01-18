class VolumeResizeProgress(object):

    def __init__(self, task_complete=False, size=None, pre_check=False):
        self.called = task_complete
        self.complete = task_complete
        self.size = size
        self.pre_check = pre_check