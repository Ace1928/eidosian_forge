from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class FakeCursorSize(Handler):

    def __init__(self, size=16):
        self.size = size

    def to_string(self) -> str:
        return 'fake_cursor_size'

    def xml_template(self) -> str:
        return '<FakeCursorSize>{{size}}</FakeCursorSize>'