import pathlib
from panel.io.mime_render import (
class Javascript:

    def __init__(self, js):
        self.js = js

    def _repr_javascript_(self):
        return self.js