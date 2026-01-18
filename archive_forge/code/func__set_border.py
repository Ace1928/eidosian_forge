from traitlets import Unicode, Instance, CaselessStrEnum, validate
from .widget import Widget, register
from .._version import __jupyter_widgets_base_version__
def _set_border(self, border):
    """
        `border` property setter. Set all 4 sides to `border` string.
        """
    for side in ['top', 'right', 'bottom', 'left']:
        setattr(self, 'border_' + side, border)