import os
from pathlib import Path
from unittest import mock
import cirq_web
class FakeWidget(cirq_web.Widget):

    def __init__(self):
        super().__init__()

    def get_client_code(self) -> str:
        return 'This is the test client code.'

    def get_widget_bundle_name(self) -> str:
        return 'testfile.txt'