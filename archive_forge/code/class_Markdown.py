import pathlib
from panel.io.mime_render import (
class Markdown:

    def __init__(self, md):
        self.md = md

    def _repr_markdown_(self):
        return self.md