from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.forms import ModelForm, modelformset_factory
from django.forms.models import BaseModelFormSet
class BaseGenericInlineFormSet(BaseModelFormSet):
    """
    A formset for generic inline objects to a parent.
    """

    def __init__(self, data=None, files=None, instance=None, save_as_new=False, prefix=None, queryset=None, **kwargs):
        opts = self.model._meta
        self.instance = instance
        self.rel_name = opts.app_label + '-' + opts.model_name + '-' + self.ct_field.name + '-' + self.ct_fk_field.name
        self.save_as_new = save_as_new
        if self.instance is None or self.instance.pk is None:
            qs = self.model._default_manager.none()
        else:
            if queryset is None:
                queryset = self.model._default_manager
            qs = queryset.filter(**{self.ct_field.name: ContentType.objects.get_for_model(self.instance, for_concrete_model=self.for_concrete_model), self.ct_fk_field.name: self.instance.pk})
        super().__init__(queryset=qs, data=data, files=files, prefix=prefix, **kwargs)

    def initial_form_count(self):
        if self.save_as_new:
            return 0
        return super().initial_form_count()

    @classmethod
    def get_default_prefix(cls):
        opts = cls.model._meta
        return opts.app_label + '-' + opts.model_name + '-' + cls.ct_field.name + '-' + cls.ct_fk_field.name

    def save_new(self, form, commit=True):
        setattr(form.instance, self.ct_field.get_attname(), ContentType.objects.get_for_model(self.instance).pk)
        setattr(form.instance, self.ct_fk_field.get_attname(), self.instance.pk)
        return form.save(commit=commit)