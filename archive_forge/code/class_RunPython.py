from django.db import router
from .base import Operation
class RunPython(Operation):
    """
    Run Python code in a context suitable for doing versioned ORM operations.
    """
    reduces_to_sql = False

    def __init__(self, code, reverse_code=None, atomic=None, hints=None, elidable=False):
        self.atomic = atomic
        if not callable(code):
            raise ValueError('RunPython must be supplied with a callable')
        self.code = code
        if reverse_code is None:
            self.reverse_code = None
        else:
            if not callable(reverse_code):
                raise ValueError('RunPython must be supplied with callable arguments')
            self.reverse_code = reverse_code
        self.hints = hints or {}
        self.elidable = elidable

    def deconstruct(self):
        kwargs = {'code': self.code}
        if self.reverse_code is not None:
            kwargs['reverse_code'] = self.reverse_code
        if self.atomic is not None:
            kwargs['atomic'] = self.atomic
        if self.hints:
            kwargs['hints'] = self.hints
        return (self.__class__.__qualname__, [], kwargs)

    @property
    def reversible(self):
        return self.reverse_code is not None

    def state_forwards(self, app_label, state):
        pass

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        from_state.clear_delayed_apps_cache()
        if router.allow_migrate(schema_editor.connection.alias, app_label, **self.hints):
            self.code(from_state.apps, schema_editor)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if self.reverse_code is None:
            raise NotImplementedError('You cannot reverse this operation')
        if router.allow_migrate(schema_editor.connection.alias, app_label, **self.hints):
            self.reverse_code(from_state.apps, schema_editor)

    def describe(self):
        return 'Raw Python operation'

    @staticmethod
    def noop(apps, schema_editor):
        return None