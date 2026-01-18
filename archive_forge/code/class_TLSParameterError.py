import requests
class TLSParameterError(DockerException):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg + '. TLS configurations should map the Docker CLI client configurations. See https://docs.docker.com/engine/articles/https/ for API details.'