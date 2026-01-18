from kivy.tests.common import GraphicUnitTest
class CallerWidget(FloatLayout):

    def __init__(self, **kwargs):
        super(CallerWidget, self).__init__(**kwargs)
        self.add_widget(UIXWidget(title='Hello World'))