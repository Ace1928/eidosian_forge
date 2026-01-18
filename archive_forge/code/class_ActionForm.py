import json
from django import forms
from django.contrib.admin.utils import (
from django.core.exceptions import ObjectDoesNotExist
from django.db.models.fields.related import (
from django.forms.utils import flatatt
from django.template.defaultfilters import capfirst, linebreaksbr
from django.urls import NoReverseMatch, reverse
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
class ActionForm(forms.Form):
    action = forms.ChoiceField(label=_('Action:'))
    select_across = forms.BooleanField(label='', required=False, initial=0, widget=forms.HiddenInput({'class': 'select-across'}))