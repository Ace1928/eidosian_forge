from itertools import chain
from django.core.exceptions import (
from django.db.models.utils import AltersData
from django.forms.fields import ChoiceField, Field
from django.forms.forms import BaseForm, DeclarativeFieldsMetaclass
from django.forms.formsets import BaseFormSet, formset_factory
from django.forms.utils import ErrorList
from django.forms.widgets import (
from django.utils.choices import BaseChoiceIterator
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
class BaseModelFormSet(BaseFormSet, AltersData):
    """
    A ``FormSet`` for editing a queryset and/or adding new objects to it.
    """
    model = None
    edit_only = False
    unique_fields = set()

    def __init__(self, data=None, files=None, auto_id='id_%s', prefix=None, queryset=None, *, initial=None, **kwargs):
        self.queryset = queryset
        self.initial_extra = initial
        super().__init__(**{'data': data, 'files': files, 'auto_id': auto_id, 'prefix': prefix, **kwargs})

    def initial_form_count(self):
        """Return the number of forms that are required in this FormSet."""
        if not self.is_bound:
            return len(self.get_queryset())
        return super().initial_form_count()

    def _existing_object(self, pk):
        if not hasattr(self, '_object_dict'):
            self._object_dict = {o.pk: o for o in self.get_queryset()}
        return self._object_dict.get(pk)

    def _get_to_python(self, field):
        """
        If the field is a related field, fetch the concrete field's (that
        is, the ultimate pointed-to field's) to_python.
        """
        while field.remote_field is not None:
            field = field.remote_field.get_related_field()
        return field.to_python

    def _construct_form(self, i, **kwargs):
        pk_required = i < self.initial_form_count()
        if pk_required:
            if self.is_bound:
                pk_key = '%s-%s' % (self.add_prefix(i), self.model._meta.pk.name)
                try:
                    pk = self.data[pk_key]
                except KeyError:
                    pass
                else:
                    to_python = self._get_to_python(self.model._meta.pk)
                    try:
                        pk = to_python(pk)
                    except ValidationError:
                        pass
                    else:
                        kwargs['instance'] = self._existing_object(pk)
            else:
                kwargs['instance'] = self.get_queryset()[i]
        elif self.initial_extra:
            try:
                kwargs['initial'] = self.initial_extra[i - self.initial_form_count()]
            except IndexError:
                pass
        form = super()._construct_form(i, **kwargs)
        if pk_required:
            form.fields[self.model._meta.pk.name].required = True
        return form

    def get_queryset(self):
        if not hasattr(self, '_queryset'):
            if self.queryset is not None:
                qs = self.queryset
            else:
                qs = self.model._default_manager.get_queryset()
            if not qs.ordered:
                qs = qs.order_by(self.model._meta.pk.name)
            self._queryset = qs
        return self._queryset

    def save_new(self, form, commit=True):
        """Save and return a new model instance for the given form."""
        return form.save(commit=commit)

    def save_existing(self, form, obj, commit=True):
        """Save and return an existing model instance for the given form."""
        return form.save(commit=commit)

    def delete_existing(self, obj, commit=True):
        """Deletes an existing model instance."""
        if commit:
            obj.delete()

    def save(self, commit=True):
        """
        Save model instances for every form, adding and changing instances
        as necessary, and return the list of instances.
        """
        if not commit:
            self.saved_forms = []

            def save_m2m():
                for form in self.saved_forms:
                    form.save_m2m()
            self.save_m2m = save_m2m
        if self.edit_only:
            return self.save_existing_objects(commit)
        else:
            return self.save_existing_objects(commit) + self.save_new_objects(commit)
    save.alters_data = True

    def clean(self):
        self.validate_unique()

    def validate_unique(self):
        all_unique_checks = set()
        all_date_checks = set()
        forms_to_delete = self.deleted_forms
        valid_forms = [form for form in self.forms if form.is_valid() and form not in forms_to_delete]
        for form in valid_forms:
            exclude = form._get_validation_exclusions()
            unique_checks, date_checks = form.instance._get_unique_checks(exclude=exclude, include_meta_constraints=True)
            all_unique_checks.update(unique_checks)
            all_date_checks.update(date_checks)
        errors = []
        for uclass, unique_check in all_unique_checks:
            seen_data = set()
            for form in valid_forms:
                row_data = (field if field in self.unique_fields else form.cleaned_data[field] for field in unique_check if field in form.cleaned_data)
                row_data = tuple((d._get_pk_val() if hasattr(d, '_get_pk_val') else tuple(d) if isinstance(d, list) else d for d in row_data))
                if row_data and None not in row_data:
                    if row_data in seen_data:
                        errors.append(self.get_unique_error_message(unique_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()], renderer=self.renderer)
                        for field in unique_check:
                            if field in form.cleaned_data:
                                del form.cleaned_data[field]
                    seen_data.add(row_data)
        for date_check in all_date_checks:
            seen_data = set()
            uclass, lookup, field, unique_for = date_check
            for form in valid_forms:
                if form.cleaned_data and form.cleaned_data[field] is not None and (form.cleaned_data[unique_for] is not None):
                    if lookup == 'date':
                        date = form.cleaned_data[unique_for]
                        date_data = (date.year, date.month, date.day)
                    else:
                        date_data = (getattr(form.cleaned_data[unique_for], lookup),)
                    data = (form.cleaned_data[field],) + date_data
                    if data in seen_data:
                        errors.append(self.get_date_error_message(date_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()], renderer=self.renderer)
                        del form.cleaned_data[field]
                    seen_data.add(data)
        if errors:
            raise ValidationError(errors)

    def get_unique_error_message(self, unique_check):
        if len(unique_check) == 1:
            return gettext('Please correct the duplicate data for %(field)s.') % {'field': unique_check[0]}
        else:
            return gettext('Please correct the duplicate data for %(field)s, which must be unique.') % {'field': get_text_list(unique_check, _('and'))}

    def get_date_error_message(self, date_check):
        return gettext('Please correct the duplicate data for %(field_name)s which must be unique for the %(lookup)s in %(date_field)s.') % {'field_name': date_check[2], 'date_field': date_check[3], 'lookup': str(date_check[1])}

    def get_form_error(self):
        return gettext('Please correct the duplicate values below.')

    def save_existing_objects(self, commit=True):
        self.changed_objects = []
        self.deleted_objects = []
        if not self.initial_forms:
            return []
        saved_instances = []
        forms_to_delete = self.deleted_forms
        for form in self.initial_forms:
            obj = form.instance
            if obj.pk is None:
                continue
            if form in forms_to_delete:
                self.deleted_objects.append(obj)
                self.delete_existing(obj, commit=commit)
            elif form.has_changed():
                self.changed_objects.append((obj, form.changed_data))
                saved_instances.append(self.save_existing(form, obj, commit=commit))
                if not commit:
                    self.saved_forms.append(form)
        return saved_instances

    def save_new_objects(self, commit=True):
        self.new_objects = []
        for form in self.extra_forms:
            if not form.has_changed():
                continue
            if self.can_delete and self._should_delete_form(form):
                continue
            self.new_objects.append(self.save_new(form, commit=commit))
            if not commit:
                self.saved_forms.append(form)
        return self.new_objects

    def add_fields(self, form, index):
        """Add a hidden field for the object's primary key."""
        from django.db.models import AutoField, ForeignKey, OneToOneField
        self._pk_field = pk = self.model._meta.pk

        def pk_is_not_editable(pk):
            return not pk.editable or (pk.auto_created or isinstance(pk, AutoField)) or (pk.remote_field and pk.remote_field.parent_link and pk_is_not_editable(pk.remote_field.model._meta.pk))
        if pk_is_not_editable(pk) or pk.name not in form.fields:
            if form.is_bound:
                pk_value = None if form.instance._state.adding else form.instance.pk
            else:
                try:
                    if index is not None:
                        pk_value = self.get_queryset()[index].pk
                    else:
                        pk_value = None
                except IndexError:
                    pk_value = None
            if isinstance(pk, (ForeignKey, OneToOneField)):
                qs = pk.remote_field.model._default_manager.get_queryset()
            else:
                qs = self.model._default_manager.get_queryset()
            qs = qs.using(form.instance._state.db)
            if form._meta.widgets:
                widget = form._meta.widgets.get(self._pk_field.name, HiddenInput)
            else:
                widget = HiddenInput
            form.fields[self._pk_field.name] = ModelChoiceField(qs, initial=pk_value, required=False, widget=widget)
        super().add_fields(form, index)