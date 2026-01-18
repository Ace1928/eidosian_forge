import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class Joystick(EventDispatcher):
    """High-level interface for joystick-like devices.  This includes a wide range
    of analog and digital joysticks, gamepads, controllers, and possibly even
    steering wheels and other input devices. There is unfortunately no easy way to
    distinguish between most of these different device types.

    For a simplified subset of Joysticks, see the :py:class:`~pyglet.input.Controller`
    interface. This covers a variety of popular game console controllers. Unlike
    Joysticks, Controllers have strictly defined layouts and inputs.

    To use a joystick, first call `open`, then in your game loop examine
    the values of `x`, `y`, and so on.  These values are normalized to the
    range [-1.0, 1.0]. 

    To receive events when the value of an axis changes, attach an 
    on_joyaxis_motion event handler to the joystick.  The :py:class:`~pyglet.input.Joystick` instance,
    axis name, and current value are passed as parameters to this event.

    To handle button events, you should attach on_joybutton_press and 
    on_joy_button_release event handlers to the joystick.  Both the :py:class:`~pyglet.input.Joystick`
    instance and the index of the changed button are passed as parameters to 
    these events.

    Alternately, you may attach event handlers to each individual button in 
    `button_controls` to receive on_press or on_release events.
    
    To use the hat switch, attach an on_joyhat_motion event handler to the
    joystick.  The handler will be called with both the hat_x and hat_y values
    whenever the value of the hat switch changes.

    The device name can be queried to get the name of the joystick.

    :Ivariables:
        `device` : `Device`
            The underlying device used by this joystick interface.
        `x` : float
            Current X (horizontal) value ranging from -1.0 (left) to 1.0
            (right).
        `y` : float
            Current y (vertical) value ranging from -1.0 (top) to 1.0
            (bottom).
        `z` : float
            Current Z value ranging from -1.0 to 1.0.  On joysticks the Z
            value is usually the throttle control.  On controllers the Z
            value is usually the secondary thumb vertical axis.
        `rx` : float
            Current rotational X value ranging from -1.0 to 1.0.
        `ry` : float
            Current rotational Y value ranging from -1.0 to 1.0.
        `rz` : float
            Current rotational Z value ranging from -1.0 to 1.0.  On joysticks
            the RZ value is usually the twist of the stick.  On game
            controllers the RZ value is usually the secondary thumb horizontal
            axis.
        `hat_x` : int
            Current hat (POV) horizontal position; one of -1 (left), 0
            (centered) or 1 (right).
        `hat_y` : int
            Current hat (POV) vertical position; one of -1 (bottom), 0
            (centered) or 1 (top).
        `buttons` : list of bool
            List of boolean values representing current states of the buttons.
            These are in order, so that button 1 has value at ``buttons[0]``,
            and so on.
        `x_control` : `AbsoluteAxis`
            Underlying control for `x` value, or ``None`` if not available.
        `y_control` : `AbsoluteAxis`
            Underlying control for `y` value, or ``None`` if not available.
        `z_control` : `AbsoluteAxis`
            Underlying control for `z` value, or ``None`` if not available.
        `rx_control` : `AbsoluteAxis`
            Underlying control for `rx` value, or ``None`` if not available.
        `ry_control` : `AbsoluteAxis`
            Underlying control for `ry` value, or ``None`` if not available.
        `rz_control` : `AbsoluteAxis`
            Underlying control for `rz` value, or ``None`` if not available.
        `hat_x_control` : `AbsoluteAxis`
            Underlying control for `hat_x` value, or ``None`` if not available.
        `hat_y_control` : `AbsoluteAxis`
            Underlying control for `hat_y` value, or ``None`` if not available.
        `button_controls` : list of `Button`
            Underlying controls for `buttons` values.
    """

    def __init__(self, device):
        self.device = device
        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0
        self.hat_x = 0
        self.hat_y = 0
        self.buttons = []
        self.x_control = None
        self.y_control = None
        self.z_control = None
        self.rx_control = None
        self.ry_control = None
        self.rz_control = None
        self.hat_x_control = None
        self.hat_y_control = None
        self.button_controls = []

        def add_axis(control):
            if not (control.min or control.max):
                warnings.warn(f"Control('{control.name}') min & max values are both 0. Skipping.")
                return
            name = control.name
            scale = 2.0 / (control.max - control.min)
            bias = -1.0 - control.min * scale
            if control.inverted:
                scale = -scale
                bias = -bias
            setattr(self, name + '_control', control)

            @control.event
            def on_change(value):
                normalized_value = value * scale + bias
                setattr(self, name, normalized_value)
                self.dispatch_event('on_joyaxis_motion', self, name, normalized_value)

        def add_button(control):
            i = len(self.buttons)
            self.buttons.append(False)
            self.button_controls.append(control)

            @control.event
            def on_change(value):
                self.buttons[i] = value

            @control.event
            def on_press():
                self.dispatch_event('on_joybutton_press', self, i)

            @control.event
            def on_release():
                self.dispatch_event('on_joybutton_release', self, i)

        def add_hat(control):
            self.hat_x_control = control
            self.hat_y_control = control

            @control.event
            def on_change(value):
                if value & 65535 == 65535:
                    self.hat_x = self.hat_y = 0
                else:
                    if control.max > 8:
                        value //= 4095
                    if 0 <= value < 8:
                        self.hat_x, self.hat_y = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))[value]
                    else:
                        self.hat_x = self.hat_y = 0
                self.dispatch_event('on_joyhat_motion', self, self.hat_x, self.hat_y)
        for ctrl in device.get_controls():
            if isinstance(ctrl, AbsoluteAxis):
                if ctrl.name in ('x', 'y', 'z', 'rx', 'ry', 'rz', 'hat_x', 'hat_y'):
                    add_axis(ctrl)
                elif ctrl.name == 'hat':
                    add_hat(ctrl)
            elif isinstance(ctrl, Button):
                add_button(ctrl)

    def open(self, window=None, exclusive=False):
        """Open the joystick device.  See `Device.open`. """
        self.device.open(window, exclusive)

    def close(self):
        """Close the joystick device.  See `Device.close`. """
        self.device.close()

    def on_joyaxis_motion(self, joystick, axis, value):
        """The value of a joystick axis changed.

        :Parameters:
            `joystick` : `Joystick`
                The joystick device whose axis changed.
            `axis` : string
                The name of the axis that changed.
            `value` : float
                The current value of the axis, normalized to [-1, 1].
        """

    def on_joybutton_press(self, joystick, button):
        """A button on the joystick was pressed.

        :Parameters:
            `joystick` : `Joystick`
                The joystick device whose button was pressed.
            `button` : int
                The index (in `button_controls`) of the button that was pressed.
        """

    def on_joybutton_release(self, joystick, button):
        """A button on the joystick was released.

        :Parameters:
            `joystick` : `Joystick`
                The joystick device whose button was released.
            `button` : int
                The index (in `button_controls`) of the button that was released.
        """

    def on_joyhat_motion(self, joystick, hat_x, hat_y):
        """The value of the joystick hat switch changed.

        :Parameters:
            `joystick` : `Joystick`
                The joystick device whose hat control changed.
            `hat_x` : int
                Current hat (POV) horizontal position; one of -1 (left), 0
                (centered) or 1 (right).
            `hat_y` : int
                Current hat (POV) vertical position; one of -1 (bottom), 0
                (centered) or 1 (top).
        """

    def __repr__(self):
        return f'Joystick(device={self.device.name})'