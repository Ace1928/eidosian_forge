import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
class Stage1(param.Parameterized):
    a = param.Number(default=5, bounds=(0, 10))
    b = param.Number(default=5, bounds=(0, 10))
    ready = param.Boolean(default=False)
    next = param.String(default=None)

    @param.output(c=param.Number)
    def output(self):
        return self.a * self.b

    @param.depends('a', 'b')
    def view(self):
        return '%s * %s = %s' % (self.a, self.b, self.output())

    def panel(self):
        return Row(self.param, self.view)