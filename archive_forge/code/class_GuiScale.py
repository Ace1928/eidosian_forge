from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class GuiScale(Handler):

    def __init__(self, gui_scale=2):
        self.gui_scale = gui_scale

    def to_string(self) -> str:
        return 'gui_scale'

    def xml_template(self) -> str:
        return '<GuiScale>{{gui_scale}}</GuiScale>'