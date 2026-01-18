from plotly.utils import _list_repr_elided
class InputDeviceState:

    def __init__(self, ctrl=None, alt=None, shift=None, meta=None, button=None, buttons=None, **_):
        self._ctrl = ctrl
        self._alt = alt
        self._meta = meta
        self._shift = shift
        self._button = button
        self._buttons = buttons

    def __repr__(self):
        return 'InputDeviceState(\n    ctrl={ctrl},\n    alt={alt},\n    shift={shift},\n    meta={meta},\n    button={button},\n    buttons={buttons})'.format(ctrl=repr(self.ctrl), alt=repr(self.alt), meta=repr(self.meta), shift=repr(self.shift), button=repr(self.button), buttons=repr(self.buttons))

    @property
    def alt(self):
        """
        Whether alt key pressed

        Returns
        -------
        bool
        """
        return self._alt

    @property
    def ctrl(self):
        """
        Whether ctrl key pressed

        Returns
        -------
        bool
        """
        return self._ctrl

    @property
    def shift(self):
        """
        Whether shift key pressed

        Returns
        -------
        bool
        """
        return self._shift

    @property
    def meta(self):
        """
        Whether meta key pressed

        Returns
        -------
        bool
        """
        return self._meta

    @property
    def button(self):
        """
        Integer code for the button that was pressed on the mouse to trigger
        the event

        - 0: Main button pressed, usually the left button or the
             un-initialized state
        - 1: Auxiliary button pressed, usually the wheel button or the middle
             button (if present)
        - 2: Secondary button pressed, usually the right button
        - 3: Fourth button, typically the Browser Back button
        - 4: Fifth button, typically the Browser Forward button

        Returns
        -------
        int
        """
        return self._button

    @property
    def buttons(self):
        """
        Integer code for which combination of buttons are pressed on the
        mouse when the event is triggered.

        -  0: No button or un-initialized
        -  1: Primary button (usually left)
        -  2: Secondary button (usually right)
        -  4: Auxilary button (usually middle or mouse wheel button)
        -  8: 4th button (typically the "Browser Back" button)
        - 16: 5th button (typically the "Browser Forward" button)

        Combinations of buttons are represented as the decimal form of the
        bitmask of the values above.

        For example, pressing both the primary (1) and auxilary (4) buttons
        will result in a code of 5

        Returns
        -------
        int
        """
        return self._buttons