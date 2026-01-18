import param
from holoviews.core.operation import Operation
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Params, Stream
class ParamClass(param.Parameterized):
    label = param.String(default='Test')

    @param.depends('label')
    def dynamic_label(self):
        return self.label + '!'