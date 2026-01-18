import copy
import datetime
from django.core.exceptions import NON_FIELD_ERRORS, ValidationError
from django.forms.fields import Field, FileField
from django.forms.utils import ErrorDict, ErrorList, RenderableFormMixin
from django.forms.widgets import Media, MediaDefiningClass
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from .renderers import get_default_renderer
def _clean_fields(self):
    for name, bf in self._bound_items():
        field = bf.field
        value = bf.initial if field.disabled else bf.data
        try:
            if isinstance(field, FileField):
                value = field.clean(value, bf.initial)
            else:
                value = field.clean(value)
            self.cleaned_data[name] = value
            if hasattr(self, 'clean_%s' % name):
                value = getattr(self, 'clean_%s' % name)()
                self.cleaned_data[name] = value
        except ValidationError as e:
            self.add_error(name, e)