from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class DoneOnDeath(Handler):
    """This should probably be in quit.py etc, but those are not implemented in Java side yet"""

    def to_string(self):
        return 'done_on_death'

    def xml_template(self) -> str:
        return '<DoneOnDeath>true</DoneOnDeath>'