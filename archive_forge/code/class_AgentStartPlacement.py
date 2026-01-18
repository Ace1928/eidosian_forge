from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class AgentStartPlacement(Handler):

    def to_string(self) -> str:
        return f'agent_start_placement({self.x}, {self.y}, {self.z}, {self.yaw}, {self.pitch})'

    def xml_template(self) -> str:
        return str('<Placement x="{{x}}" y="{{y}}" z="{{z}}" yaw="{{yaw}}" pitch="{{pitch}}"/>')

    def __init__(self, x, y, z, yaw=0.0, pitch=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch