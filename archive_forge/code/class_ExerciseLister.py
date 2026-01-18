import weakref
from unittest import mock
from cliff import lister
from cliff.tests import base
class ExerciseLister(lister.Lister):
    data = [('a', 'A'), ('b', 'B'), ('c', 'A')]

    def _load_formatter_plugins(self):
        return {'test': FauxFormatter()}

    def take_action(self, parsed_args):
        return (parsed_args.columns, self.data)