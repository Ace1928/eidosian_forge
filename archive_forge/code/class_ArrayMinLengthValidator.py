from django.core.exceptions import ValidationError
from django.core.validators import (
from django.utils.deconstruct import deconstructible
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
class ArrayMinLengthValidator(MinLengthValidator):
    message = ngettext_lazy('List contains %(show_value)d item, it should contain no fewer than %(limit_value)d.', 'List contains %(show_value)d items, it should contain no fewer than %(limit_value)d.', 'show_value')