import datetime
import importlib
import os
import sys
from django.apps import apps
from django.core.management.base import OutputWrapper
from django.db.models import NOT_PROVIDED
from django.utils import timezone
from django.utils.version import get_docs_version
from .loader import MigrationLoader
class InteractiveMigrationQuestioner(MigrationQuestioner):

    def __init__(self, defaults=None, specified_apps=None, dry_run=None, prompt_output=None):
        super().__init__(defaults=defaults, specified_apps=specified_apps, dry_run=dry_run)
        self.prompt_output = prompt_output or OutputWrapper(sys.stdout)

    def _boolean_input(self, question, default=None):
        self.prompt_output.write(f'{question} ', ending='')
        result = input()
        if not result and default is not None:
            return default
        while not result or result[0].lower() not in 'yn':
            self.prompt_output.write('Please answer yes or no: ', ending='')
            result = input()
        return result[0].lower() == 'y'

    def _choice_input(self, question, choices):
        self.prompt_output.write(f'{question}')
        for i, choice in enumerate(choices):
            self.prompt_output.write(' %s) %s' % (i + 1, choice))
        self.prompt_output.write('Select an option: ', ending='')
        result = input()
        while True:
            try:
                value = int(result)
            except ValueError:
                pass
            else:
                if 0 < value <= len(choices):
                    return value
            self.prompt_output.write('Please select a valid option: ', ending='')
            result = input()

    def _ask_default(self, default=''):
        """
        Prompt for a default value.

        The ``default`` argument allows providing a custom default value (as a
        string) which will be shown to the user and used as the return value
        if the user doesn't provide any other input.
        """
        self.prompt_output.write('Please enter the default value as valid Python.')
        if default:
            self.prompt_output.write(f"Accept the default '{default}' by pressing 'Enter' or provide another value.")
        self.prompt_output.write('The datetime and django.utils.timezone modules are available, so it is possible to provide e.g. timezone.now as a value.')
        self.prompt_output.write("Type 'exit' to exit this prompt")
        while True:
            if default:
                prompt = '[default: {}] >>> '.format(default)
            else:
                prompt = '>>> '
            self.prompt_output.write(prompt, ending='')
            code = input()
            if not code and default:
                code = default
            if not code:
                self.prompt_output.write("Please enter some code, or 'exit' (without quotes) to exit.")
            elif code == 'exit':
                sys.exit(1)
            else:
                try:
                    return eval(code, {}, {'datetime': datetime, 'timezone': timezone})
                except (SyntaxError, NameError) as e:
                    self.prompt_output.write('Invalid input: %s' % e)

    def ask_not_null_addition(self, field_name, model_name):
        """Adding a NOT NULL field to a model."""
        if not self.dry_run:
            choice = self._choice_input(f"It is impossible to add a non-nullable field '{field_name}' to {model_name} without specifying a default. This is because the database needs something to populate existing rows.\nPlease select a fix:", ['Provide a one-off default now (will be set on all existing rows with a null value for this column)', 'Quit and manually define a default value in models.py.'])
            if choice == 2:
                sys.exit(3)
            else:
                return self._ask_default()
        return None

    def ask_not_null_alteration(self, field_name, model_name):
        """Changing a NULL field to NOT NULL."""
        if not self.dry_run:
            choice = self._choice_input(f"It is impossible to change a nullable field '{field_name}' on {model_name} to non-nullable without providing a default. This is because the database needs something to populate existing rows.\nPlease select a fix:", ['Provide a one-off default now (will be set on all existing rows with a null value for this column)', 'Ignore for now. Existing rows that contain NULL values will have to be handled manually, for example with a RunPython or RunSQL operation.', 'Quit and manually define a default value in models.py.'])
            if choice == 2:
                return NOT_PROVIDED
            elif choice == 3:
                sys.exit(3)
            else:
                return self._ask_default()
        return None

    def ask_rename(self, model_name, old_name, new_name, field_instance):
        """Was this field really renamed?"""
        msg = 'Was %s.%s renamed to %s.%s (a %s)? [y/N]'
        return self._boolean_input(msg % (model_name, old_name, model_name, new_name, field_instance.__class__.__name__), False)

    def ask_rename_model(self, old_model_state, new_model_state):
        """Was this model really renamed?"""
        msg = 'Was the model %s.%s renamed to %s? [y/N]'
        return self._boolean_input(msg % (old_model_state.app_label, old_model_state.name, new_model_state.name), False)

    def ask_merge(self, app_label):
        return self._boolean_input('\nMerging will only work if the operations printed above do not conflict\n' + 'with each other (working on different fields or models)\n' + 'Should these migration branches be merged? [y/N]', False)

    def ask_auto_now_add_addition(self, field_name, model_name):
        """Adding an auto_now_add field to a model."""
        if not self.dry_run:
            choice = self._choice_input(f"It is impossible to add the field '{field_name}' with 'auto_now_add=True' to {model_name} without providing a default. This is because the database needs something to populate existing rows.\n", ['Provide a one-off default now which will be set on all existing rows', 'Quit and manually define a default value in models.py.'])
            if choice == 2:
                sys.exit(3)
            else:
                return self._ask_default(default='timezone.now')
        return None

    def ask_unique_callable_default_addition(self, field_name, model_name):
        """Adding a unique field with a callable default."""
        if not self.dry_run:
            version = get_docs_version()
            choice = self._choice_input(f'Callable default on unique field {model_name}.{field_name} will not generate unique values upon migrating.\nPlease choose how to proceed:\n', [f'Continue making this migration as the first step in writing a manual migration to generate unique values described here: https://docs.djangoproject.com/en/{version}/howto/writing-migrations/#migrations-that-add-unique-fields.', 'Quit and edit field options in models.py.'])
            if choice == 2:
                sys.exit(3)
        return None