class VolumeDetachProgress(object):

    def __init__(self, srv_id, vol_id, attach_id, task_complete=False):
        self.called = task_complete
        self.cinder_complete = task_complete
        self.nova_complete = task_complete
        self.srv_id = srv_id
        self.vol_id = vol_id
        self.attach_id = attach_id