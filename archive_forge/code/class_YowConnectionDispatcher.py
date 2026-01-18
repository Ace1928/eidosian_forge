class YowConnectionDispatcher(object):

    def __init__(self, connectionCallbacks):
        assert isinstance(connectionCallbacks, ConnectionCallbacks)
        self.connectionCallbacks = connectionCallbacks

    def connect(self, host):
        pass

    def disconnect(self):
        pass

    def sendData(self, data):
        pass