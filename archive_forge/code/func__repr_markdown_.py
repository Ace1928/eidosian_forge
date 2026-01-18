import pathlib
from panel.io.mime_render import (
def _repr_markdown_(self):
    return self.md