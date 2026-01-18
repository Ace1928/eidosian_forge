import requests
class BuildError(DockerException):

    def __init__(self, reason, build_log):
        super().__init__(reason)
        self.msg = reason
        self.build_log = build_log