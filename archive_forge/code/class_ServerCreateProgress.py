class ServerCreateProgress(object):

    def __init__(self, server_id, complete=False):
        self.complete = complete
        self.server_id = server_id