from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget import register
from .trait_types import Color, NumberFormat
class TagsInputBase(DescriptionWidget, ValueWidget, CoreWidget):
    _model_name = Unicode('TagsInputBaseModel').tag(sync=True)
    value = List().tag(sync=True)
    placeholder = Unicode('\u200b').tag(sync=True)
    allowed_tags = List().tag(sync=True)
    allow_duplicates = Bool(True).tag(sync=True)

    @validate('value')
    def _validate_value(self, proposal):
        if '' in proposal['value']:
            raise TraitError('The value of a TagsInput widget cannot contain blank strings')
        if len(self.allowed_tags) == 0:
            return proposal['value']
        for tag_value in proposal['value']:
            if tag_value not in self.allowed_tags:
                raise TraitError('Tag value {} is not allowed, allowed tags are {}'.format(tag_value, self.allowed_tags))
        return proposal['value']