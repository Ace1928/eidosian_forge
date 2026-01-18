import copy
import json
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import CASCADE, UUIDField
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.html import smart_urlquote
from django.utils.http import urlencode
from django.utils.text import Truncator
from django.utils.translation import get_language
from django.utils.translation import gettext as _
class AdminURLFieldWidget(forms.URLInput):
    template_name = 'admin/widgets/url.html'

    def __init__(self, attrs=None, validator_class=URLValidator):
        super().__init__(attrs={'class': 'vURLField', **(attrs or {})})
        self.validator = validator_class()

    def get_context(self, name, value, attrs):
        try:
            self.validator(value if value else '')
            url_valid = True
        except ValidationError:
            url_valid = False
        context = super().get_context(name, value, attrs)
        context['current_label'] = _('Currently:')
        context['change_label'] = _('Change:')
        context['widget']['href'] = smart_urlquote(context['widget']['value']) if value else ''
        context['url_valid'] = url_valid
        return context