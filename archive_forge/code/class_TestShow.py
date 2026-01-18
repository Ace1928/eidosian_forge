import weakref
from unittest import mock
from cliff import show
from cliff.tests import base
class TestShow(base.TestBase):

    def test_formatter_args(self):
        app = mock.Mock()
        test_show = ExerciseShowOne(app, [])
        parsed_args = mock.Mock()
        parsed_args.columns = ('Col1', 'Col2')
        parsed_args.formatter = 'test'
        test_show.run(parsed_args)
        f = test_show._formatter_plugins['test']
        self.assertEqual(1, len(f.args))
        args = f.args[0]
        self.assertEqual(list(parsed_args.columns), args[0])
        data = list(args[1])
        self.assertEqual([('a', 'A'), ('b', 'B')], data)

    def test_dict2columns(self):
        app = mock.Mock()
        test_show = ExerciseShowOne(app, [])
        d = {'a': 'A', 'b': 'B', 'c': 'C'}
        expected = [('a', 'b', 'c'), ('A', 'B', 'C')]
        actual = list(test_show.dict2columns(d))
        self.assertEqual(expected, actual)

    def test_no_exist_column(self):
        test_show = ExerciseShowOne(mock.Mock(), [])
        parsed_args = mock.Mock()
        parsed_args.columns = ('no_exist_column',)
        parsed_args.formatter = 'test'
        with mock.patch.object(test_show, 'take_action') as mock_take_action:
            mock_take_action.return_value = (('Col1', 'Col2', 'Col3'), [])
            self.assertRaises(ValueError, test_show.run, parsed_args)