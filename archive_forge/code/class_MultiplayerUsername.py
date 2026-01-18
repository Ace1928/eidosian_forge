from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class MultiplayerUsername(Handler):

    def __init__(self, name: str):
        self.name = name

    def to_string(self):
        return 'multiplayer_username'

    def xml_template(self) -> str:
        return '<MultiplayerUsername>{{ name }}</MultiplayerUsername>'