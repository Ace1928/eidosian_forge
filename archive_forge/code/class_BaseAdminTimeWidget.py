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
class BaseAdminTimeWidget(forms.TimeInput):

    class Media:
        js = ['admin/js/calendar.js', 'admin/js/admin/DateTimeShortcuts.js']

    def __init__(self, attrs=None, format=None):
        attrs = {'class': 'vTimeField', 'size': '8', **(attrs or {})}
        super().__init__(attrs=attrs, format=format)