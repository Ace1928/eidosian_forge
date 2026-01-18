from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class CreateModel(ModelOperation):
    """Create a model's table."""
    serialization_expand_args = ['fields', 'options', 'managers']

    def __init__(self, name, fields, options=None, bases=None, managers=None):
        self.fields = fields
        self.options = options or {}
        self.bases = bases or (models.Model,)
        self.managers = managers or []
        super().__init__(name)
        _check_for_duplicates('fields', (name for name, _ in self.fields))
        _check_for_duplicates('bases', (base._meta.label_lower if hasattr(base, '_meta') else base.lower() if isinstance(base, str) else base for base in self.bases))
        _check_for_duplicates('managers', (name for name, _ in self.managers))

    def deconstruct(self):
        kwargs = {'name': self.name, 'fields': self.fields}
        if self.options:
            kwargs['options'] = self.options
        if self.bases and self.bases != (models.Model,):
            kwargs['bases'] = self.bases
        if self.managers and self.managers != [('objects', models.Manager())]:
            kwargs['managers'] = self.managers
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.add_model(ModelState(app_label, self.name, list(self.fields), dict(self.options), tuple(self.bases), list(self.managers)))

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.create_model(model)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.delete_model(model)

    def describe(self):
        return 'Create %smodel %s' % ('proxy ' if self.options.get('proxy', False) else '', self.name)

    @property
    def migration_name_fragment(self):
        return self.name_lower

    def references_model(self, name, app_label):
        name_lower = name.lower()
        if name_lower == self.name_lower:
            return True
        reference_model_tuple = (app_label, name_lower)
        for base in self.bases:
            if base is not models.Model and isinstance(base, (models.base.ModelBase, str)) and (resolve_relation(base, app_label) == reference_model_tuple):
                return True
        for _name, field in self.fields:
            if field_references((app_label, self.name_lower), field, reference_model_tuple):
                return True
        return False

    def reduce(self, operation, app_label):
        if isinstance(operation, DeleteModel) and self.name_lower == operation.name_lower and (not self.options.get('proxy', False)):
            return []
        elif isinstance(operation, RenameModel) and self.name_lower == operation.old_name_lower:
            return [CreateModel(operation.new_name, fields=self.fields, options=self.options, bases=self.bases, managers=self.managers)]
        elif isinstance(operation, AlterModelOptions) and self.name_lower == operation.name_lower:
            options = {**self.options, **operation.options}
            for key in operation.ALTER_OPTION_KEYS:
                if key not in operation.options:
                    options.pop(key, None)
            return [CreateModel(self.name, fields=self.fields, options=options, bases=self.bases, managers=self.managers)]
        elif isinstance(operation, AlterModelManagers) and self.name_lower == operation.name_lower:
            return [CreateModel(self.name, fields=self.fields, options=self.options, bases=self.bases, managers=operation.managers)]
        elif isinstance(operation, AlterTogetherOptionOperation) and self.name_lower == operation.name_lower:
            return [CreateModel(self.name, fields=self.fields, options={**self.options, **{operation.option_name: operation.option_value}}, bases=self.bases, managers=self.managers)]
        elif isinstance(operation, AlterOrderWithRespectTo) and self.name_lower == operation.name_lower:
            return [CreateModel(self.name, fields=self.fields, options={**self.options, 'order_with_respect_to': operation.order_with_respect_to}, bases=self.bases, managers=self.managers)]
        elif isinstance(operation, FieldOperation) and self.name_lower == operation.model_name_lower:
            if isinstance(operation, AddField):
                return [CreateModel(self.name, fields=self.fields + [(operation.name, operation.field)], options=self.options, bases=self.bases, managers=self.managers)]
            elif isinstance(operation, AlterField):
                return [CreateModel(self.name, fields=[(n, operation.field if n == operation.name else v) for n, v in self.fields], options=self.options, bases=self.bases, managers=self.managers)]
            elif isinstance(operation, RemoveField):
                options = self.options.copy()
                for option_name in ('unique_together', 'index_together'):
                    option = options.pop(option_name, None)
                    if option:
                        option = set(filter(bool, (tuple((f for f in fields if f != operation.name_lower)) for fields in option)))
                        if option:
                            options[option_name] = option
                order_with_respect_to = options.get('order_with_respect_to')
                if order_with_respect_to == operation.name_lower:
                    del options['order_with_respect_to']
                return [CreateModel(self.name, fields=[(n, v) for n, v in self.fields if n.lower() != operation.name_lower], options=options, bases=self.bases, managers=self.managers)]
            elif isinstance(operation, RenameField):
                options = self.options.copy()
                for option_name in ('unique_together', 'index_together'):
                    option = options.get(option_name)
                    if option:
                        options[option_name] = {tuple((operation.new_name if f == operation.old_name else f for f in fields)) for fields in option}
                order_with_respect_to = options.get('order_with_respect_to')
                if order_with_respect_to == operation.old_name:
                    options['order_with_respect_to'] = operation.new_name
                return [CreateModel(self.name, fields=[(operation.new_name if n == operation.old_name else n, v) for n, v in self.fields], options=options, bases=self.bases, managers=self.managers)]
        elif isinstance(operation, IndexOperation) and self.name_lower == operation.model_name_lower:
            if isinstance(operation, AddIndex):
                return [CreateModel(self.name, fields=self.fields, options={**self.options, 'indexes': [*self.options.get('indexes', []), operation.index]}, bases=self.bases, managers=self.managers)]
            elif isinstance(operation, RemoveIndex):
                options_indexes = [index for index in self.options.get('indexes', []) if index.name != operation.name]
                return [CreateModel(self.name, fields=self.fields, options={**self.options, 'indexes': options_indexes}, bases=self.bases, managers=self.managers)]
            elif isinstance(operation, RenameIndex) and operation.old_fields:
                options_index_together = {fields for fields in self.options.get('index_together', []) if fields != operation.old_fields}
                if options_index_together:
                    self.options['index_together'] = options_index_together
                else:
                    self.options.pop('index_together', None)
                return [CreateModel(self.name, fields=self.fields, options={**self.options, 'indexes': [*self.options.get('indexes', []), models.Index(fields=operation.old_fields, name=operation.new_name)]}, bases=self.bases, managers=self.managers)]
        return super().reduce(operation, app_label)