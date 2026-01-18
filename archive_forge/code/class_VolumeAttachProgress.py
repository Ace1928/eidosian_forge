class VolumeAttachProgress(object):

    def __init__(self, srv_id, vol_id, device, task_complete=False):
        self.called = task_complete
        self.complete = task_complete
        self.srv_id = srv_id
        self.vol_id = vol_id
        self.device = device