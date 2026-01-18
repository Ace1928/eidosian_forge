from kivy.tests import async_run, UnitKivyApp
from math import isclose
class ModalButton(Button):
    """ button used as root widget to test touch. """
    modal = None

    def on_touch_down(self, touch):
        """ touch down event handler. """
        assert self.modal._window is None
        assert not self.modal._is_open
        return super(ModalButton, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        """ touch move event handler. """
        assert self.modal._window is None
        assert not self.modal._is_open
        return super(ModalButton, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        """ touch up event handler. """
        assert self.modal._window is None
        assert not self.modal._is_open
        return super(ModalButton, self).on_touch_up(touch)