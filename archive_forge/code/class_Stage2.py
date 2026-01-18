import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
class Stage2(param.Parameterized):
    c = param.Number(default=5, precedence=-1, bounds=(0, None))
    exp = param.Number(default=0.1, bounds=(0, 3))

    @param.depends('c', 'exp')
    def view(self):
        return '%s^%s=%.3f' % (self.c, self.exp, self.c ** self.exp)

    def panel(self):
        return Row(self.param, self.view)