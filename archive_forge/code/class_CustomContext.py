import zmq
class CustomContext(zmq.Context):
    extra_arg: str
    _socket_class = CustomSocket

    def __init__(self, extra_arg: str='x'):
        super().__init__()
        self.extra_arg = extra_arg