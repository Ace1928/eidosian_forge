from traitlets import Unicode, Instance, CaselessStrEnum, validate
from .widget import Widget, register
from .._version import __jupyter_widgets_base_version__
class LayoutTraitType(Instance):
    klass = Layout

    def validate(self, obj, value):
        if isinstance(value, dict):
            return super().validate(obj, self.klass(**value))
        else:
            return super().validate(obj, value)