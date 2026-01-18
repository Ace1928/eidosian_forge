from .widget import Widget, register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Unicode, Tuple, Instance, TraitError
class WidgetTraitTuple(Tuple):
    """Traitlet for validating a single (Widget, 'trait_name') pair"""
    info_text = "A (Widget, 'trait_name') pair"

    def __init__(self, **kwargs):
        super().__init__(Instance(Widget), Unicode(), **kwargs)
        if 'default_value' not in kwargs and (not kwargs.get('allow_none', False)):
            self.default_args = ()

    def validate_elements(self, obj, value):
        value = super().validate_elements(obj, value)
        widget, trait_name = value
        trait = widget.traits().get(trait_name)
        trait_repr = '{}.{}'.format(widget.__class__.__name__, trait_name)
        if trait is None:
            raise TypeError('No such trait: %s' % trait_repr)
        elif not trait.metadata.get('sync'):
            raise TypeError('%s cannot be synced' % trait_repr)
        return value