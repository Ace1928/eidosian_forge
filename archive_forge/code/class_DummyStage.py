import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
class DummyStage(param.Parameterized):

    def panel(self):
        return 'foo'