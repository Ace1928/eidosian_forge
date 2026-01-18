from kivy.tests import async_run, UnitKivyApp
class RootButton(Button):
    dropdown = None

    def on_touch_down(self, touch):
        assert self.dropdown.attach_to is None
        return super(RootButton, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        assert self.dropdown.attach_to is None
        return super(RootButton, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        assert self.dropdown.attach_to is None
        return super(RootButton, self).on_touch_up(touch)