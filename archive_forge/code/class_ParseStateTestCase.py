from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
class ParseStateTestCase(test_base.BaseTestCase):

    def test_init(self):
        state = _parser.ParseState()
        self.assertEqual([], state.tokens)
        self.assertEqual([], state.values)

    @mock.patch.object(_parser.ParseState, 'reducers', [(['tok1'], 'meth')])
    @mock.patch.object(_parser.ParseState, 'meth', create=True)
    def test_reduce_none(self, mock_meth):
        state = _parser.ParseState()
        state.tokens = ['tok2']
        state.values = ['val2']
        state.reduce()
        self.assertEqual(['tok2'], state.tokens)
        self.assertEqual(['val2'], state.values)
        self.assertFalse(mock_meth.called)

    @mock.patch.object(_parser.ParseState, 'reducers', [(['tok1', 'tok2'], 'meth')])
    @mock.patch.object(_parser.ParseState, 'meth', create=True)
    def test_reduce_short(self, mock_meth):
        state = _parser.ParseState()
        state.tokens = ['tok1']
        state.values = ['val1']
        state.reduce()
        self.assertEqual(['tok1'], state.tokens)
        self.assertEqual(['val1'], state.values)
        self.assertFalse(mock_meth.called)

    @mock.patch.object(_parser.ParseState, 'reducers', [(['tok1', 'tok2'], 'meth')])
    @mock.patch.object(_parser.ParseState, 'meth', create=True, return_value=[('tok3', 'val3')])
    def test_reduce_one(self, mock_meth):
        state = _parser.ParseState()
        state.tokens = ['tok1', 'tok2']
        state.values = ['val1', 'val2']
        state.reduce()
        self.assertEqual(['tok3'], state.tokens)
        self.assertEqual(['val3'], state.values)
        mock_meth.assert_called_once_with('val1', 'val2')

    @mock.patch.object(_parser.ParseState, 'reducers', [(['tok1', 'tok4'], 'meth2'), (['tok2', 'tok3'], 'meth1')])
    @mock.patch.object(_parser.ParseState, 'meth1', create=True, return_value=[('tok4', 'val4')])
    @mock.patch.object(_parser.ParseState, 'meth2', create=True, return_value=[('tok5', 'val5')])
    def test_reduce_two(self, mock_meth2, mock_meth1):
        state = _parser.ParseState()
        state.tokens = ['tok1', 'tok2', 'tok3']
        state.values = ['val1', 'val2', 'val3']
        state.reduce()
        self.assertEqual(['tok5'], state.tokens)
        self.assertEqual(['val5'], state.values)
        mock_meth1.assert_called_once_with('val2', 'val3')
        mock_meth2.assert_called_once_with('val1', 'val4')

    @mock.patch.object(_parser.ParseState, 'reducers', [(['tok1', 'tok2'], 'meth')])
    @mock.patch.object(_parser.ParseState, 'meth', create=True, return_value=[('tok3', 'val3'), ('tok4', 'val4')])
    def test_reduce_multi(self, mock_meth):
        state = _parser.ParseState()
        state.tokens = ['tok1', 'tok2']
        state.values = ['val1', 'val2']
        state.reduce()
        self.assertEqual(['tok3', 'tok4'], state.tokens)
        self.assertEqual(['val3', 'val4'], state.values)
        mock_meth.assert_called_once_with('val1', 'val2')

    def test_shift(self):
        state = _parser.ParseState()
        with mock.patch.object(_parser.ParseState, 'reduce') as mock_reduce:
            state.shift('token', 'value')
            self.assertEqual(['token'], state.tokens)
            self.assertEqual(['value'], state.values)
            mock_reduce.assert_called_once_with()

    def test_result_empty(self):
        state = _parser.ParseState()
        self.assertRaises(ValueError, lambda: state.result)

    def test_result_unreduced(self):
        state = _parser.ParseState()
        state.tokens = ['tok1', 'tok2']
        state.values = ['val1', 'val2']
        self.assertRaises(ValueError, lambda: state.result)

    def test_result(self):
        state = _parser.ParseState()
        state.tokens = ['token']
        state.values = ['value']
        self.assertEqual('value', state.result)

    def test_wrap_check(self):
        state = _parser.ParseState()
        result = state._wrap_check('(', 'the_check', ')')
        self.assertEqual([('check', 'the_check')], result)

    @mock.patch.object(_checks, 'AndCheck', lambda x: x)
    def test_make_and_expr(self):
        state = _parser.ParseState()
        result = state._make_and_expr('check1', 'and', 'check2')
        self.assertEqual([('and_expr', ['check1', 'check2'])], result)

    def test_extend_and_expr(self):
        state = _parser.ParseState()
        mock_expr = mock.Mock()
        mock_expr.add_check.return_value = 'newcheck'
        result = state._extend_and_expr(mock_expr, 'and', 'check')
        self.assertEqual([('and_expr', 'newcheck')], result)
        mock_expr.add_check.assert_called_once_with('check')

    @mock.patch.object(_checks, 'OrCheck', lambda x: x)
    def test_make_or_expr(self):
        state = _parser.ParseState()
        result = state._make_or_expr('check1', 'or', 'check2')
        self.assertEqual([('or_expr', ['check1', 'check2'])], result)

    def test_extend_or_expr(self):
        state = _parser.ParseState()
        mock_expr = mock.Mock()
        mock_expr.add_check.return_value = 'newcheck'
        result = state._extend_or_expr(mock_expr, 'or', 'check')
        self.assertEqual([('or_expr', 'newcheck')], result)
        mock_expr.add_check.assert_called_once_with('check')

    @mock.patch.object(_checks, 'NotCheck', lambda x: 'not %s' % x)
    def test_make_not_expr(self):
        state = _parser.ParseState()
        result = state._make_not_expr('not', 'check')
        self.assertEqual([('check', 'not check')], result)