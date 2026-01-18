from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.renderers import get_default_renderer
from django.forms.utils import ErrorList, RenderableFormMixin
from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
class BaseFormSet(RenderableFormMixin):
    """
    A collection of instances of the same Form class.
    """
    deletion_widget = CheckboxInput
    ordering_widget = NumberInput
    default_error_messages = {'missing_management_form': _('ManagementForm data is missing or has been tampered with. Missing fields: %(field_names)s. You may need to file a bug report if the issue persists.'), 'too_many_forms': ngettext_lazy('Please submit at most %(num)d form.', 'Please submit at most %(num)d forms.', 'num'), 'too_few_forms': ngettext_lazy('Please submit at least %(num)d form.', 'Please submit at least %(num)d forms.', 'num')}
    template_name_div = 'django/forms/formsets/div.html'
    template_name_p = 'django/forms/formsets/p.html'
    template_name_table = 'django/forms/formsets/table.html'
    template_name_ul = 'django/forms/formsets/ul.html'

    def __init__(self, data=None, files=None, auto_id='id_%s', prefix=None, initial=None, error_class=ErrorList, form_kwargs=None, error_messages=None):
        self.is_bound = data is not None or files is not None
        self.prefix = prefix or self.get_default_prefix()
        self.auto_id = auto_id
        self.data = data or {}
        self.files = files or {}
        self.initial = initial
        self.form_kwargs = form_kwargs or {}
        self.error_class = error_class
        self._errors = None
        self._non_form_errors = None
        self.form_renderer = self.renderer
        self.renderer = self.renderer or get_default_renderer()
        messages = {}
        for cls in reversed(type(self).__mro__):
            messages.update(getattr(cls, 'default_error_messages', {}))
        if error_messages is not None:
            messages.update(error_messages)
        self.error_messages = messages

    def __iter__(self):
        """Yield the forms in the order they should be rendered."""
        return iter(self.forms)

    def __getitem__(self, index):
        """Return the form at the given index, based on the rendering order."""
        return self.forms[index]

    def __len__(self):
        return len(self.forms)

    def __bool__(self):
        """
        Return True since all formsets have a management form which is not
        included in the length.
        """
        return True

    def __repr__(self):
        if self._errors is None:
            is_valid = 'Unknown'
        else:
            is_valid = self.is_bound and (not self._non_form_errors) and (not any((form_errors for form_errors in self._errors)))
        return '<%s: bound=%s valid=%s total_forms=%s>' % (self.__class__.__qualname__, self.is_bound, is_valid, self.total_form_count())

    @cached_property
    def management_form(self):
        """Return the ManagementForm instance for this FormSet."""
        if self.is_bound:
            form = ManagementForm(self.data, auto_id=self.auto_id, prefix=self.prefix, renderer=self.renderer)
            form.full_clean()
        else:
            form = ManagementForm(auto_id=self.auto_id, prefix=self.prefix, initial={TOTAL_FORM_COUNT: self.total_form_count(), INITIAL_FORM_COUNT: self.initial_form_count(), MIN_NUM_FORM_COUNT: self.min_num, MAX_NUM_FORM_COUNT: self.max_num}, renderer=self.renderer)
        return form

    def total_form_count(self):
        """Return the total number of forms in this FormSet."""
        if self.is_bound:
            return min(self.management_form.cleaned_data[TOTAL_FORM_COUNT], self.absolute_max)
        else:
            initial_forms = self.initial_form_count()
            total_forms = max(initial_forms, self.min_num) + self.extra
            if initial_forms > self.max_num >= 0:
                total_forms = initial_forms
            elif total_forms > self.max_num >= 0:
                total_forms = self.max_num
        return total_forms

    def initial_form_count(self):
        """Return the number of forms that are required in this FormSet."""
        if self.is_bound:
            return self.management_form.cleaned_data[INITIAL_FORM_COUNT]
        else:
            initial_forms = len(self.initial) if self.initial else 0
        return initial_forms

    @cached_property
    def forms(self):
        """Instantiate forms at first property access."""
        return [self._construct_form(i, **self.get_form_kwargs(i)) for i in range(self.total_form_count())]

    def get_form_kwargs(self, index):
        """
        Return additional keyword arguments for each individual formset form.

        index will be None if the form being constructed is a new empty
        form.
        """
        return self.form_kwargs.copy()

    def _construct_form(self, i, **kwargs):
        """Instantiate and return the i-th form instance in a formset."""
        defaults = {'auto_id': self.auto_id, 'prefix': self.add_prefix(i), 'error_class': self.error_class, 'use_required_attribute': False, 'renderer': self.form_renderer}
        if self.is_bound:
            defaults['data'] = self.data
            defaults['files'] = self.files
        if self.initial and 'initial' not in kwargs:
            try:
                defaults['initial'] = self.initial[i]
            except IndexError:
                pass
        if i >= self.initial_form_count() and i >= self.min_num:
            defaults['empty_permitted'] = True
        defaults.update(kwargs)
        form = self.form(**defaults)
        self.add_fields(form, i)
        return form

    @property
    def initial_forms(self):
        """Return a list of all the initial forms in this formset."""
        return self.forms[:self.initial_form_count()]

    @property
    def extra_forms(self):
        """Return a list of all the extra forms in this formset."""
        return self.forms[self.initial_form_count():]

    @property
    def empty_form(self):
        form_kwargs = {**self.get_form_kwargs(None), 'auto_id': self.auto_id, 'prefix': self.add_prefix('__prefix__'), 'empty_permitted': True, 'use_required_attribute': False, 'renderer': self.form_renderer}
        form = self.form(**form_kwargs)
        self.add_fields(form, None)
        return form

    @property
    def cleaned_data(self):
        """
        Return a list of form.cleaned_data dicts for every form in self.forms.
        """
        if not self.is_valid():
            raise AttributeError("'%s' object has no attribute 'cleaned_data'" % self.__class__.__name__)
        return [form.cleaned_data for form in self.forms]

    @property
    def deleted_forms(self):
        """Return a list of forms that have been marked for deletion."""
        if not self.is_valid() or not self.can_delete:
            return []
        if not hasattr(self, '_deleted_form_indexes'):
            self._deleted_form_indexes = []
            for i, form in enumerate(self.forms):
                if i >= self.initial_form_count() and (not form.has_changed()):
                    continue
                if self._should_delete_form(form):
                    self._deleted_form_indexes.append(i)
        return [self.forms[i] for i in self._deleted_form_indexes]

    @property
    def ordered_forms(self):
        """
        Return a list of form in the order specified by the incoming data.
        Raise an AttributeError if ordering is not allowed.
        """
        if not self.is_valid() or not self.can_order:
            raise AttributeError("'%s' object has no attribute 'ordered_forms'" % self.__class__.__name__)
        if not hasattr(self, '_ordering'):
            self._ordering = []
            for i, form in enumerate(self.forms):
                if i >= self.initial_form_count() and (not form.has_changed()):
                    continue
                if self.can_delete and self._should_delete_form(form):
                    continue
                self._ordering.append((i, form.cleaned_data[ORDERING_FIELD_NAME]))

            def compare_ordering_key(k):
                if k[1] is None:
                    return (1, 0)
                return (0, k[1])
            self._ordering.sort(key=compare_ordering_key)
        return [self.forms[i[0]] for i in self._ordering]

    @classmethod
    def get_default_prefix(cls):
        return 'form'

    @classmethod
    def get_deletion_widget(cls):
        return cls.deletion_widget

    @classmethod
    def get_ordering_widget(cls):
        return cls.ordering_widget

    def non_form_errors(self):
        """
        Return an ErrorList of errors that aren't associated with a particular
        form -- i.e., from formset.clean(). Return an empty ErrorList if there
        are none.
        """
        if self._non_form_errors is None:
            self.full_clean()
        return self._non_form_errors

    @property
    def errors(self):
        """Return a list of form.errors for every form in self.forms."""
        if self._errors is None:
            self.full_clean()
        return self._errors

    def total_error_count(self):
        """Return the number of errors across all forms in the formset."""
        return len(self.non_form_errors()) + sum((len(form_errors) for form_errors in self.errors))

    def _should_delete_form(self, form):
        """Return whether or not the form was marked for deletion."""
        return form.cleaned_data.get(DELETION_FIELD_NAME, False)

    def is_valid(self):
        """Return True if every form in self.forms is valid."""
        if not self.is_bound:
            return False
        self.errors
        forms_valid = all([form.is_valid() for form in self.forms if not (self.can_delete and self._should_delete_form(form))])
        return forms_valid and (not self.non_form_errors())

    def full_clean(self):
        """
        Clean all of self.data and populate self._errors and
        self._non_form_errors.
        """
        self._errors = []
        self._non_form_errors = self.error_class(error_class='nonform', renderer=self.renderer)
        empty_forms_count = 0
        if not self.is_bound:
            return
        if not self.management_form.is_valid():
            error = ValidationError(self.error_messages['missing_management_form'], params={'field_names': ', '.join((self.management_form.add_prefix(field_name) for field_name in self.management_form.errors))}, code='missing_management_form')
            self._non_form_errors.append(error)
        for i, form in enumerate(self.forms):
            if not form.has_changed() and i >= self.initial_form_count():
                empty_forms_count += 1
            form_errors = form.errors
            if self.can_delete and self._should_delete_form(form):
                continue
            self._errors.append(form_errors)
        try:
            if self.validate_max and self.total_form_count() - len(self.deleted_forms) > self.max_num or self.management_form.cleaned_data[TOTAL_FORM_COUNT] > self.absolute_max:
                raise ValidationError(self.error_messages['too_many_forms'] % {'num': self.max_num}, code='too_many_forms')
            if self.validate_min and self.total_form_count() - len(self.deleted_forms) - empty_forms_count < self.min_num:
                raise ValidationError(self.error_messages['too_few_forms'] % {'num': self.min_num}, code='too_few_forms')
            self.clean()
        except ValidationError as e:
            self._non_form_errors = self.error_class(e.error_list, error_class='nonform', renderer=self.renderer)

    def clean(self):
        """
        Hook for doing any extra formset-wide cleaning after Form.clean() has
        been called on every form. Any ValidationError raised by this method
        will not be associated with a particular form; it will be accessible
        via formset.non_form_errors()
        """
        pass

    def has_changed(self):
        """Return True if data in any form differs from initial."""
        return any((form.has_changed() for form in self))

    def add_fields(self, form, index):
        """A hook for adding extra fields on to each form instance."""
        initial_form_count = self.initial_form_count()
        if self.can_order:
            if index is not None and index < initial_form_count:
                form.fields[ORDERING_FIELD_NAME] = IntegerField(label=_('Order'), initial=index + 1, required=False, widget=self.get_ordering_widget())
            else:
                form.fields[ORDERING_FIELD_NAME] = IntegerField(label=_('Order'), required=False, widget=self.get_ordering_widget())
        if self.can_delete and (self.can_delete_extra or (index is not None and index < initial_form_count)):
            form.fields[DELETION_FIELD_NAME] = BooleanField(label=_('Delete'), required=False, widget=self.get_deletion_widget())

    def add_prefix(self, index):
        return '%s-%s' % (self.prefix, index)

    def is_multipart(self):
        """
        Return True if the formset needs to be multipart, i.e. it
        has FileInput, or False otherwise.
        """
        if self.forms:
            return self.forms[0].is_multipart()
        else:
            return self.empty_form.is_multipart()

    @property
    def media(self):
        if self.forms:
            return self.forms[0].media
        else:
            return self.empty_form.media

    @property
    def template_name(self):
        return self.renderer.formset_template_name

    def get_context(self):
        return {'formset': self}