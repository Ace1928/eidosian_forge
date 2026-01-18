from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class LoadWorldAgentStart(Handler):

    def __init__(self, filename):
        self.filename = filename

    def to_string(self) -> str:
        return 'load_world_file'

    def xml_template(self) -> str:
        _filename = None
        if callable(self.filename):
            _filename = self.filename()
        elif isinstance(self.filename, str):
            _filename = self.filename
        if _filename is not None:
            return f'<LoadWorldFile>{_filename}</LoadWorldFile>'
        return ''