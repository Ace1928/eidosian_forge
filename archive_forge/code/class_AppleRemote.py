import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class AppleRemote(EventDispatcher):
    """High-level interface for Apple remote control.

    This interface provides access to the 6 button controls on the remote.
    Pressing and holding certain buttons on the remote is interpreted as
    a separate control.

    :Ivariables:
        `device` : `Device`
            The underlying device used by this interface.
        `left_control` : `Button`
            Button control for the left (prev) button.
        `left_hold_control` : `Button`
            Button control for holding the left button (rewind).
        `right_control` : `Button`
            Button control for the right (next) button.
        `right_hold_control` : `Button`
            Button control for holding the right button (fast forward).
        `up_control` : `Button`
            Button control for the up (volume increase) button.
        `down_control` : `Button`
            Button control for the down (volume decrease) button.
        `select_control` : `Button`
            Button control for the select (play/pause) button.
        `select_hold_control` : `Button`
            Button control for holding the select button.
        `menu_control` : `Button`
            Button control for the menu button.
        `menu_hold_control` : `Button`
            Button control for holding the menu button.
    """

    def __init__(self, device):

        def add_button(control):
            setattr(self, control.name + '_control', control)

            @control.event
            def on_press():
                self.dispatch_event('on_button_press', control.name)

            @control.event
            def on_release():
                self.dispatch_event('on_button_release', control.name)
        self.device = device
        for control in device.get_controls():
            if control.name in ('left', 'left_hold', 'right', 'right_hold', 'up', 'down', 'menu', 'select', 'menu_hold', 'select_hold'):
                add_button(control)

    def open(self, window=None, exclusive=False):
        """Open the device.  See `Device.open`. """
        self.device.open(window, exclusive)

    def close(self):
        """Close the device.  See `Device.close`. """
        self.device.close()

    def on_button_press(self, button):
        """A button on the remote was pressed.

        Only the 'up' and 'down' buttons will generate an event when the
        button is first pressed.  All other buttons on the remote will wait
        until the button is released and then send both the press and release
        events at the same time.

        :Parameters:
            `button` : unicode
                The name of the button that was pressed. The valid names are
                'up', 'down', 'left', 'right', 'left_hold', 'right_hold',
                'menu', 'menu_hold', 'select', and 'select_hold'
                
        :event:
        """

    def on_button_release(self, button):
        """A button on the remote was released.

        The 'select_hold' and 'menu_hold' button release events are sent
        immediately after the corresponding press events regardless of
        whether the user has released the button.

        :Parameters:
            `button` : str
                The name of the button that was released. The valid names are
                'up', 'down', 'left', 'right', 'left_hold', 'right_hold',
                'menu', 'menu_hold', 'select', and 'select_hold'

        :event:
        """