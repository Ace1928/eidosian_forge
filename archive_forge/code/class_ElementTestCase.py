from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
class ElementTestCase(ComparisonTestCase):

    def setUp(self):
        self.el1 = Element('data1')
        self.el2 = Element('data2')
        self.el3 = Element('data3')
        self.el4 = Element('data4', group='ValA')
        self.el5 = Element('data5', group='ValB')
        self.el6 = Element('data6', label='LabelA')
        self.el7 = Element('data7', group='ValA', label='LabelA')
        self.el8 = Element('data8', group='ValA', label='LabelB')

    def test_element_init(self):
        Element('data1')