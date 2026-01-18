import unittest
class TestWidget(Widget):
    source = StringProperty('')
    source2 = StringProperty('')
    source3 = StringProperty('')
    can_edit = BooleanProperty(False)

    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super(TestWidget, self).__init__(**kwargs)

    def on_release(self):
        pass