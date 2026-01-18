from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class FOVSetting(Handler):

    def __init__(self, fov_setting=130.0):
        self.fov_setting = fov_setting

    def to_string(self) -> str:
        return 'fov_setting'

    def xml_template(self) -> str:
        return '<FOVSetting>{{fov_setting}}</FOVSetting>'