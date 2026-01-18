from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class SpawnInVillage(Handler):

    def to_string(self):
        return 'start_in_village'

    def xml_template(self) -> str:
        return '<SpawnInVillage>true</SpawnInVillage>'