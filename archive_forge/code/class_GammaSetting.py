from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class GammaSetting(Handler):

    def __init__(self, gamma_setting=2.0):
        self.gamma_setting = gamma_setting

    def to_string(self) -> str:
        return 'gamma_setting'

    def xml_template(self) -> str:
        return '<GammaSetting>{{gamma_setting}}</GammaSetting>'