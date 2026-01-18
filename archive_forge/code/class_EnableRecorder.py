from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class EnableRecorder(Handler):

    def to_string(self) -> str:
        return 'enable_recorder'

    def xml_template(self) -> str:
        return '<EnableRecorder>true</EnableRecorder>'