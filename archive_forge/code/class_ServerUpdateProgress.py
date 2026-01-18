class ServerUpdateProgress(UpdateProgressBase):

    def __init__(self, server_id, handler, complete=False, called=False, handler_extra=None, checker_extra=None):
        super(ServerUpdateProgress, self).__init__(server_id, handler, complete=complete, called=called, handler_extra=handler_extra, checker_extra=checker_extra)
        self.server_id = server_id