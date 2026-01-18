class ServerDeleteProgress(object):

    def __init__(self, server_id, image_id=None, image_complete=True):
        self.server_id = server_id
        self.image_id = image_id
        self.image_complete = image_complete