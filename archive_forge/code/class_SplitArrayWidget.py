import copy
from itertools import chain
from django import forms
from django.contrib.postgres.validators import (
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from ..utils import prefix_validation_error
class SplitArrayWidget(forms.Widget):
    template_name = 'postgres/widgets/split_array.html'

    def __init__(self, widget, size, **kwargs):
        self.widget = widget() if isinstance(widget, type) else widget
        self.size = size
        super().__init__(**kwargs)

    @property
    def is_hidden(self):
        return self.widget.is_hidden

    def value_from_datadict(self, data, files, name):
        return [self.widget.value_from_datadict(data, files, '%s_%s' % (name, index)) for index in range(self.size)]

    def value_omitted_from_data(self, data, files, name):
        return all((self.widget.value_omitted_from_data(data, files, '%s_%s' % (name, index)) for index in range(self.size)))

    def id_for_label(self, id_):
        if id_:
            id_ += '_0'
        return id_

    def get_context(self, name, value, attrs=None):
        attrs = {} if attrs is None else attrs
        context = super().get_context(name, value, attrs)
        if self.is_localized:
            self.widget.is_localized = self.is_localized
        value = value or []
        context['widget']['subwidgets'] = []
        final_attrs = self.build_attrs(attrs)
        id_ = final_attrs.get('id')
        for i in range(max(len(value), self.size)):
            try:
                widget_value = value[i]
            except IndexError:
                widget_value = None
            if id_:
                final_attrs = {**final_attrs, 'id': '%s_%s' % (id_, i)}
            context['widget']['subwidgets'].append(self.widget.get_context(name + '_%s' % i, widget_value, final_attrs)['widget'])
        return context

    @property
    def media(self):
        return self.widget.media

    def __deepcopy__(self, memo):
        obj = super().__deepcopy__(memo)
        obj.widget = copy.deepcopy(self.widget)
        return obj

    @property
    def needs_multipart_form(self):
        return self.widget.needs_multipart_form