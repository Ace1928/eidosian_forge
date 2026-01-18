class VolumeDeleteProgress(object):

    def __init__(self, task_complete=False):
        self.backup = {'called': task_complete, 'complete': task_complete}
        self.delete = {'called': task_complete, 'complete': task_complete}
        self.backup_id = None