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
class AutocompleteMixin:
    """
    Select widget mixin that loads options from AutocompleteJsonView via AJAX.

    Renders the necessary data attributes for select2 and adds the static form
    media.
    """
    url_name = '%s:autocomplete'

    def __init__(self, field, admin_site, attrs=None, choices=(), using=None):
        self.field = field
        self.admin_site = admin_site
        self.db = using
        self.choices = choices
        self.attrs = {} if attrs is None else attrs.copy()
        self.i18n_name = get_select2_language()

    def get_url(self):
        return reverse(self.url_name % self.admin_site.name)

    def build_attrs(self, base_attrs, extra_attrs=None):
        """
        Set select2's AJAX attributes.

        Attributes can be set using the html5 data attribute.
        Nested attributes require a double dash as per
        https://select2.org/configuration/data-attributes#nested-subkey-options
        """
        attrs = super().build_attrs(base_attrs, extra_attrs=extra_attrs)
        attrs.setdefault('class', '')
        attrs.update({'data-ajax--cache': 'true', 'data-ajax--delay': 250, 'data-ajax--type': 'GET', 'data-ajax--url': self.get_url(), 'data-app-label': self.field.model._meta.app_label, 'data-model-name': self.field.model._meta.model_name, 'data-field-name': self.field.name, 'data-theme': 'admin-autocomplete', 'data-allow-clear': json.dumps(not self.is_required), 'data-placeholder': '', 'lang': self.i18n_name, 'class': attrs['class'] + (' ' if attrs['class'] else '') + 'admin-autocomplete'})
        return attrs

    def optgroups(self, name, value, attr=None):
        """Return selected options based on the ModelChoiceIterator."""
        default = (None, [], 0)
        groups = [default]
        has_selected = False
        selected_choices = {str(v) for v in value if str(v) not in self.choices.field.empty_values}
        if not self.is_required and (not self.allow_multiple_selected):
            default[1].append(self.create_option(name, '', '', False, 0))
        remote_model_opts = self.field.remote_field.model._meta
        to_field_name = getattr(self.field.remote_field, 'field_name', remote_model_opts.pk.attname)
        to_field_name = remote_model_opts.get_field(to_field_name).attname
        choices = ((getattr(obj, to_field_name), self.choices.field.label_from_instance(obj)) for obj in self.choices.queryset.using(self.db).filter(**{'%s__in' % to_field_name: selected_choices}))
        for option_value, option_label in choices:
            selected = str(option_value) in value and (has_selected is False or self.allow_multiple_selected)
            has_selected |= selected
            index = len(default[1])
            subgroup = default[1]
            subgroup.append(self.create_option(name, option_value, option_label, selected_choices, index))
        return groups

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % self.i18n_name,) if self.i18n_name else ()
        return forms.Media(js=('admin/js/vendor/jquery/jquery%s.js' % extra, 'admin/js/vendor/select2/select2.full%s.js' % extra) + i18n_file + ('admin/js/jquery.init.js', 'admin/js/autocomplete.js'), css={'screen': ('admin/css/vendor/select2/select2%s.css' % extra, 'admin/css/autocomplete.css')})