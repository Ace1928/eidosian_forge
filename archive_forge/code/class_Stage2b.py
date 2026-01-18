import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
class Stage2b(param.Parameterized):
    c = param.Number(default=5, precedence=-1, bounds=(0, None))
    root = param.Parameter(default=0.1)

    @param.depends('c', 'root')
    def view(self):
        return '%s^-%s=%.3f' % (self.c, self.root, self.c ** (-self.root))

    def panel(self):
        return Row(self.param, self.view)