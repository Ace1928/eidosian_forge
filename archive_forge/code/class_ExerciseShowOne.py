import weakref
from unittest import mock
from cliff import show
from cliff.tests import base
class ExerciseShowOne(show.ShowOne):

    def _load_formatter_plugins(self):
        return {'test': FauxFormatter()}

    def take_action(self, parsed_args):
        return (parsed_args.columns, [('a', 'A'), ('b', 'B')])