import json
from django.contrib.postgres import lookups
from django.contrib.postgres.forms import SimpleArrayField
from django.contrib.postgres.validators import ArrayMaxLengthValidator
from django.core import checks, exceptions
from django.db.models import Field, Func, IntegerField, Transform, Value
from django.db.models.fields.mixins import CheckFieldDefaultMixin
from django.db.models.lookups import Exact, In
from django.utils.translation import gettext_lazy as _
from ..utils import prefix_validation_error
from .utils import AttributeSetter
class ArrayField(CheckFieldDefaultMixin, Field):
    empty_strings_allowed = False
    default_error_messages = {'item_invalid': _('Item %(nth)s in the array did not validate:'), 'nested_array_mismatch': _('Nested arrays must have the same length.')}
    _default_hint = ('list', '[]')

    def __init__(self, base_field, size=None, **kwargs):
        self.base_field = base_field
        self.db_collation = getattr(self.base_field, 'db_collation', None)
        self.size = size
        if self.size:
            self.default_validators = [*self.default_validators, ArrayMaxLengthValidator(self.size)]
        if hasattr(self.base_field, 'from_db_value'):
            self.from_db_value = self._from_db_value
        super().__init__(**kwargs)

    @property
    def model(self):
        try:
            return self.__dict__['model']
        except KeyError:
            raise AttributeError("'%s' object has no attribute 'model'" % self.__class__.__name__)

    @model.setter
    def model(self, model):
        self.__dict__['model'] = model
        self.base_field.model = model

    @classmethod
    def _choices_is_value(cls, value):
        return isinstance(value, (list, tuple)) or super()._choices_is_value(value)

    def check(self, **kwargs):
        errors = super().check(**kwargs)
        if self.base_field.remote_field:
            errors.append(checks.Error('Base field for array cannot be a related field.', obj=self, id='postgres.E002'))
        else:
            base_checks = self.base_field.check()
            if base_checks:
                error_messages = '\n    '.join(('%s (%s)' % (base_check.msg, base_check.id) for base_check in base_checks if isinstance(base_check, checks.Error)))
                if error_messages:
                    errors.append(checks.Error('Base field for array has errors:\n    %s' % error_messages, obj=self, id='postgres.E001'))
                warning_messages = '\n    '.join(('%s (%s)' % (base_check.msg, base_check.id) for base_check in base_checks if isinstance(base_check, checks.Warning)))
                if warning_messages:
                    errors.append(checks.Warning('Base field for array has warnings:\n    %s' % warning_messages, obj=self, id='postgres.W004'))
        return errors

    def set_attributes_from_name(self, name):
        super().set_attributes_from_name(name)
        self.base_field.set_attributes_from_name(name)

    @property
    def description(self):
        return 'Array of %s' % self.base_field.description

    def db_type(self, connection):
        size = self.size or ''
        return '%s[%s]' % (self.base_field.db_type(connection), size)

    def cast_db_type(self, connection):
        size = self.size or ''
        return '%s[%s]' % (self.base_field.cast_db_type(connection), size)

    def db_parameters(self, connection):
        db_params = super().db_parameters(connection)
        db_params['collation'] = self.db_collation
        return db_params

    def get_placeholder(self, value, compiler, connection):
        return '%s::{}'.format(self.db_type(connection))

    def get_db_prep_value(self, value, connection, prepared=False):
        if isinstance(value, (list, tuple)):
            return [self.base_field.get_db_prep_value(i, connection, prepared=False) for i in value]
        return value

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if path == 'django.contrib.postgres.fields.array.ArrayField':
            path = 'django.contrib.postgres.fields.ArrayField'
        kwargs.update({'base_field': self.base_field.clone(), 'size': self.size})
        return (name, path, args, kwargs)

    def to_python(self, value):
        if isinstance(value, str):
            vals = json.loads(value)
            value = [self.base_field.to_python(val) for val in vals]
        return value

    def _from_db_value(self, value, expression, connection):
        if value is None:
            return value
        return [self.base_field.from_db_value(item, expression, connection) for item in value]

    def value_to_string(self, obj):
        values = []
        vals = self.value_from_object(obj)
        base_field = self.base_field
        for val in vals:
            if val is None:
                values.append(None)
            else:
                obj = AttributeSetter(base_field.attname, val)
                values.append(base_field.value_to_string(obj))
        return json.dumps(values)

    def get_transform(self, name):
        transform = super().get_transform(name)
        if transform:
            return transform
        if '_' not in name:
            try:
                index = int(name)
            except ValueError:
                pass
            else:
                index += 1
                return IndexTransformFactory(index, self.base_field)
        try:
            start, end = name.split('_')
            start = int(start) + 1
            end = int(end)
        except ValueError:
            pass
        else:
            return SliceTransformFactory(start, end)

    def validate(self, value, model_instance):
        super().validate(value, model_instance)
        for index, part in enumerate(value):
            try:
                self.base_field.validate(part, model_instance)
            except exceptions.ValidationError as error:
                raise prefix_validation_error(error, prefix=self.error_messages['item_invalid'], code='item_invalid', params={'nth': index + 1})
        if isinstance(self.base_field, ArrayField):
            if len({len(i) for i in value}) > 1:
                raise exceptions.ValidationError(self.error_messages['nested_array_mismatch'], code='nested_array_mismatch')

    def run_validators(self, value):
        super().run_validators(value)
        for index, part in enumerate(value):
            try:
                self.base_field.run_validators(part)
            except exceptions.ValidationError as error:
                raise prefix_validation_error(error, prefix=self.error_messages['item_invalid'], code='item_invalid', params={'nth': index + 1})

    def formfield(self, **kwargs):
        return super().formfield(**{'form_class': SimpleArrayField, 'base_field': self.base_field.formfield(), 'max_length': self.size, **kwargs})