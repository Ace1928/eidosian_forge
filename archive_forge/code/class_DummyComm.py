from ipywidgets import Widget
import ipywidgets.widgets.widget
import ipykernel.comm
class DummyComm:
    comm_id = 'a-b-c-d'
    kernel = 'Truthy'

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.messages = []

    def open(self, *args, **kwargs):
        pass

    def on_msg(self, *args, **kwargs):
        pass

    def send(self, *args, **kwargs):
        self.messages.append((args, kwargs))

    def close(self, *args, **kwargs):
        pass