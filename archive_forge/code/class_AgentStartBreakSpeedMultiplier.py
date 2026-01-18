from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class AgentStartBreakSpeedMultiplier(Handler):

    def to_string(self) -> str:
        return f'agent_start_break_speed_multiplier({self.multiplier})'

    def xml_template(self) -> str:
        return str('<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>')

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier