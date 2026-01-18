import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
class TestControllerArguments(PecanTestCase):

    @property
    def app_(self):

        class RootController(object):

            @expose()
            def index(self, req, resp, id):
                return 'index: %s' % id

            @expose()
            def multiple(self, req, resp, one, two):
                return 'multiple: %s, %s' % (one, two)

            @expose()
            def optional(self, req, resp, id=None):
                return 'optional: %s' % str(id)

            @expose()
            def multiple_optional(self, req, resp, one=None, two=None, three=None):
                return 'multiple_optional: %s, %s, %s' % (one, two, three)

            @expose()
            def variable_args(self, req, resp, *args):
                return 'variable_args: %s' % ', '.join(args)

            @expose()
            def variable_kwargs(self, req, resp, **kwargs):
                data = ['%s=%s' % (key, kwargs[key]) for key in sorted(kwargs.keys())]
                return 'variable_kwargs: %s' % ', '.join(data)

            @expose()
            def variable_all(self, req, resp, *args, **kwargs):
                data = ['%s=%s' % (key, kwargs[key]) for key in sorted(kwargs.keys())]
                return 'variable_all: %s' % ', '.join(list(args) + data)

            @expose()
            def eater(self, req, resp, id, dummy=None, *args, **kwargs):
                data = ['%s=%s' % (key, kwargs[key]) for key in sorted(kwargs.keys())]
                return 'eater: %s, %s, %s' % (id, dummy, ', '.join(list(args) + data))

            @expose()
            def _route(self, args, request):
                if hasattr(self, args[0]):
                    return (getattr(self, args[0]), args[1:])
                else:
                    return (self.index, args)
        return TestApp(Pecan(RootController(), use_context_locals=False))

    def test_required_argument(self):
        try:
            r = self.app_.get('/')
            assert r.status_int != 200
        except Exception as ex:
            assert type(ex) == TypeError
            assert ex.args[0] in ('index() takes exactly 2 arguments (1 given)', "index() missing 1 required positional argument: 'id'", "TestControllerArguments.app_.<locals>.RootController.index() missing 1 required positional argument: 'id'")

    def test_single_argument(self):
        r = self.app_.get('/1')
        assert r.status_int == 200
        assert r.body == b'index: 1'

    def test_single_argument_with_encoded_url(self):
        r = self.app_.get('/This%20is%20a%20test%21')
        assert r.status_int == 200
        assert r.body == b'index: This is a test!'

    def test_two_arguments(self):
        r = self.app_.get('/1/dummy', status=404)
        assert r.status_int == 404

    def test_keyword_argument(self):
        r = self.app_.get('/?id=2')
        assert r.status_int == 200
        assert r.body == b'index: 2'

    def test_keyword_argument_with_encoded_url(self):
        r = self.app_.get('/?id=This%20is%20a%20test%21')
        assert r.status_int == 200
        assert r.body == b'index: This is a test!'

    def test_argument_and_keyword_argument(self):
        r = self.app_.get('/3?id=three')
        assert r.status_int == 200
        assert r.body == b'index: 3'

    def test_encoded_argument_and_keyword_argument(self):
        r = self.app_.get('/This%20is%20a%20test%21?id=three')
        assert r.status_int == 200
        assert r.body == b'index: This is a test!'

    def test_explicit_kwargs(self):
        r = self.app_.post('/', {'id': '4'})
        assert r.status_int == 200
        assert r.body == b'index: 4'

    def test_path_with_explicit_kwargs(self):
        r = self.app_.post('/4', {'id': 'four'})
        assert r.status_int == 200
        assert r.body == b'index: 4'

    def test_multiple_kwargs(self):
        r = self.app_.get('/?id=5&dummy=dummy')
        assert r.status_int == 200
        assert r.body == b'index: 5'

    def test_kwargs_from_root(self):
        r = self.app_.post('/', {'id': '6', 'dummy': 'dummy'})
        assert r.status_int == 200
        assert r.body == b'index: 6'

    def test_multiple_positional_arguments(self):
        r = self.app_.get('/multiple/one/two')
        assert r.status_int == 200
        assert r.body == b'multiple: one, two'

    def test_multiple_positional_arguments_with_url_encode(self):
        r = self.app_.get('/multiple/One%20/Two%21')
        assert r.status_int == 200
        assert r.body == b'multiple: One , Two!'

    def test_multiple_positional_arguments_with_kwargs(self):
        r = self.app_.get('/multiple?one=three&two=four')
        assert r.status_int == 200
        assert r.body == b'multiple: three, four'

    def test_multiple_positional_arguments_with_url_encoded_kwargs(self):
        r = self.app_.get('/multiple?one=Three%20&two=Four%20%21')
        assert r.status_int == 200
        assert r.body == b'multiple: Three , Four !'

    def test_positional_args_with_dictionary_kwargs(self):
        r = self.app_.post('/multiple', {'one': 'five', 'two': 'six'})
        assert r.status_int == 200
        assert r.body == b'multiple: five, six'

    def test_positional_args_with_url_encoded_dictionary_kwargs(self):
        r = self.app_.post('/multiple', {'one': 'Five%20', 'two': 'Six%20%21'})
        assert r.status_int == 200
        assert r.body == b'multiple: Five%20, Six%20%21'

    def test_optional_arg(self):
        r = self.app_.get('/optional')
        assert r.status_int == 200
        assert r.body == b'optional: None'

    def test_multiple_optional(self):
        r = self.app_.get('/optional/1')
        assert r.status_int == 200
        assert r.body == b'optional: 1'

    def test_multiple_optional_url_encoded(self):
        r = self.app_.get('/optional/Some%20Number')
        assert r.status_int == 200
        assert r.body == b'optional: Some Number'

    def test_multiple_optional_missing(self):
        r = self.app_.get('/optional/2/dummy', status=404)
        assert r.status_int == 404

    def test_multiple_with_kwargs(self):
        r = self.app_.get('/optional?id=2')
        assert r.status_int == 200
        assert r.body == b'optional: 2'

    def test_multiple_with_url_encoded_kwargs(self):
        r = self.app_.get('/optional?id=Some%20Number')
        assert r.status_int == 200
        assert r.body == b'optional: Some Number'

    def test_multiple_args_with_url_encoded_kwargs(self):
        r = self.app_.get('/optional/3?id=three')
        assert r.status_int == 200
        assert r.body == b'optional: 3'

    def test_url_encoded_positional_args(self):
        r = self.app_.get('/optional/Some%20Number?id=three')
        assert r.status_int == 200
        assert r.body == b'optional: Some Number'

    def test_optional_arg_with_kwargs(self):
        r = self.app_.post('/optional', {'id': '4'})
        assert r.status_int == 200
        assert r.body == b'optional: 4'

    def test_optional_arg_with_url_encoded_kwargs(self):
        r = self.app_.post('/optional', {'id': 'Some%20Number'})
        assert r.status_int == 200
        assert r.body == b'optional: Some%20Number'

    def test_multiple_positional_arguments_with_dictionary_kwargs(self):
        r = self.app_.post('/optional/5', {'id': 'five'})
        assert r.status_int == 200
        assert r.body == b'optional: 5'

    def test_multiple_positional_url_encoded_arguments_with_kwargs(self):
        r = self.app_.post('/optional/Some%20Number', {'id': 'five'})
        assert r.status_int == 200
        assert r.body == b'optional: Some Number'

    def test_optional_arg_with_multiple_kwargs(self):
        r = self.app_.get('/optional?id=6&dummy=dummy')
        assert r.status_int == 200
        assert r.body == b'optional: 6'

    def test_optional_arg_with_multiple_url_encoded_kwargs(self):
        r = self.app_.get('/optional?id=Some%20Number&dummy=dummy')
        assert r.status_int == 200
        assert r.body == b'optional: Some Number'

    def test_optional_arg_with_multiple_dictionary_kwargs(self):
        r = self.app_.post('/optional', {'id': '7', 'dummy': 'dummy'})
        assert r.status_int == 200
        assert r.body == b'optional: 7'

    def test_optional_arg_with_multiple_url_encoded_dictionary_kwargs(self):
        r = self.app_.post('/optional', {'id': 'Some%20Number', 'dummy': 'dummy'})
        assert r.status_int == 200
        assert r.body == b'optional: Some%20Number'

    def test_multiple_optional_positional_args(self):
        r = self.app_.get('/multiple_optional')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: None, None, None'

    def test_multiple_optional_positional_args_one_arg(self):
        r = self.app_.get('/multiple_optional/1')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, None, None'

    def test_multiple_optional_positional_args_one_url_encoded_arg(self):
        r = self.app_.get('/multiple_optional/One%21')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One!, None, None'

    def test_multiple_optional_positional_args_all_args(self):
        r = self.app_.get('/multiple_optional/1/2/3')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, 2, 3'

    def test_multiple_optional_positional_args_all_url_encoded_args(self):
        r = self.app_.get('/multiple_optional/One%21/Two%21/Three%21')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One!, Two!, Three!'

    def test_multiple_optional_positional_args_too_many_args(self):
        r = self.app_.get('/multiple_optional/1/2/3/dummy', status=404)
        assert r.status_int == 404

    def test_multiple_optional_positional_args_with_kwargs(self):
        r = self.app_.get('/multiple_optional?one=1')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, None, None'

    def test_multiple_optional_positional_args_with_url_encoded_kwargs(self):
        r = self.app_.get('/multiple_optional?one=One%21')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One!, None, None'

    def test_multiple_optional_positional_args_with_string_kwargs(self):
        r = self.app_.get('/multiple_optional/1?one=one')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, None, None'

    def test_multiple_optional_positional_args_with_encoded_str_kwargs(self):
        r = self.app_.get('/multiple_optional/One%21?one=one')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One!, None, None'

    def test_multiple_optional_positional_args_with_dict_kwargs(self):
        r = self.app_.post('/multiple_optional', {'one': '1'})
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, None, None'

    def test_multiple_optional_positional_args_with_encoded_dict_kwargs(self):
        r = self.app_.post('/multiple_optional', {'one': 'One%21'})
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One%21, None, None'

    def test_multiple_optional_positional_args_and_dict_kwargs(self):
        r = self.app_.post('/multiple_optional/1', {'one': 'one'})
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, None, None'

    def test_multiple_optional_encoded_positional_args_and_dict_kwargs(self):
        r = self.app_.post('/multiple_optional/One%21', {'one': 'one'})
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One!, None, None'

    def test_multiple_optional_args_with_multiple_kwargs(self):
        r = self.app_.get('/multiple_optional?one=1&two=2&three=3&four=4')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, 2, 3'

    def test_multiple_optional_args_with_multiple_encoded_kwargs(self):
        r = self.app_.get('/multiple_optional?one=One%21&two=Two%21&three=Three%21&four=4')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One!, Two!, Three!'

    def test_multiple_optional_args_with_multiple_dict_kwargs(self):
        r = self.app_.post('/multiple_optional', {'one': '1', 'two': '2', 'three': '3', 'four': '4'})
        assert r.status_int == 200
        assert r.body == b'multiple_optional: 1, 2, 3'

    def test_multiple_optional_args_with_multiple_encoded_dict_kwargs(self):
        r = self.app_.post('/multiple_optional', {'one': 'One%21', 'two': 'Two%21', 'three': 'Three%21', 'four': '4'})
        assert r.status_int == 200
        assert r.body == b'multiple_optional: One%21, Two%21, Three%21'

    def test_multiple_optional_args_with_last_kwarg(self):
        r = self.app_.get('/multiple_optional?three=3')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: None, None, 3'

    def test_multiple_optional_args_with_last_encoded_kwarg(self):
        r = self.app_.get('/multiple_optional?three=Three%21')
        assert r.status_int == 200
        assert r.body == b'multiple_optional: None, None, Three!'

    def test_multiple_optional_args_with_middle_arg(self):
        r = self.app_.get('/multiple_optional', {'two': '2'})
        assert r.status_int == 200
        assert r.body == b'multiple_optional: None, 2, None'

    def test_variable_args(self):
        r = self.app_.get('/variable_args')
        assert r.status_int == 200
        assert r.body == b'variable_args: '

    def test_multiple_variable_args(self):
        r = self.app_.get('/variable_args/1/dummy')
        assert r.status_int == 200
        assert r.body == b'variable_args: 1, dummy'

    def test_multiple_encoded_variable_args(self):
        r = self.app_.get('/variable_args/Testing%20One%20Two/Three%21')
        assert r.status_int == 200
        assert r.body == b'variable_args: Testing One Two, Three!'

    def test_variable_args_with_kwargs(self):
        r = self.app_.get('/variable_args?id=2&dummy=dummy')
        assert r.status_int == 200
        assert r.body == b'variable_args: '

    def test_variable_args_with_dict_kwargs(self):
        r = self.app_.post('/variable_args', {'id': '3', 'dummy': 'dummy'})
        assert r.status_int == 200
        assert r.body == b'variable_args: '

    def test_variable_kwargs(self):
        r = self.app_.get('/variable_kwargs')
        assert r.status_int == 200
        assert r.body == b'variable_kwargs: '

    def test_multiple_variable_kwargs(self):
        r = self.app_.get('/variable_kwargs/1/dummy', status=404)
        assert r.status_int == 404

    def test_multiple_variable_kwargs_with_explicit_kwargs(self):
        r = self.app_.get('/variable_kwargs?id=2&dummy=dummy')
        assert r.status_int == 200
        assert r.body == b'variable_kwargs: dummy=dummy, id=2'

    def test_multiple_variable_kwargs_with_explicit_encoded_kwargs(self):
        r = self.app_.get('/variable_kwargs?id=Two%21&dummy=This%20is%20a%20test')
        assert r.status_int == 200
        assert r.body == b'variable_kwargs: dummy=This is a test, id=Two!'

    def test_multiple_variable_kwargs_with_dict_kwargs(self):
        r = self.app_.post('/variable_kwargs', {'id': '3', 'dummy': 'dummy'})
        assert r.status_int == 200
        assert r.body == b'variable_kwargs: dummy=dummy, id=3'

    def test_multiple_variable_kwargs_with_encoded_dict_kwargs(self):
        r = self.app_.post('/variable_kwargs', {'id': 'Three%21', 'dummy': 'This%20is%20a%20test'})
        assert r.status_int == 200
        result = b'variable_kwargs: dummy=This%20is%20a%20test, id=Three%21'
        assert r.body == result

    def test_variable_all(self):
        r = self.app_.get('/variable_all')
        assert r.status_int == 200
        assert r.body == b'variable_all: '

    def test_variable_all_with_one_extra(self):
        r = self.app_.get('/variable_all/1')
        assert r.status_int == 200
        assert r.body == b'variable_all: 1'

    def test_variable_all_with_two_extras(self):
        r = self.app_.get('/variable_all/2/dummy')
        assert r.status_int == 200
        assert r.body == b'variable_all: 2, dummy'

    def test_variable_mixed(self):
        r = self.app_.get('/variable_all/3?month=1&day=12')
        assert r.status_int == 200
        assert r.body == b'variable_all: 3, day=12, month=1'

    def test_variable_mixed_explicit(self):
        r = self.app_.get('/variable_all/4?id=four&month=1&day=12')
        assert r.status_int == 200
        assert r.body == b'variable_all: 4, day=12, id=four, month=1'

    def test_variable_post(self):
        r = self.app_.post('/variable_all/5/dummy')
        assert r.status_int == 200
        assert r.body == b'variable_all: 5, dummy'

    def test_variable_post_with_kwargs(self):
        r = self.app_.post('/variable_all/6', {'month': '1', 'day': '12'})
        assert r.status_int == 200
        assert r.body == b'variable_all: 6, day=12, month=1'

    def test_variable_post_mixed(self):
        r = self.app_.post('/variable_all/7', {'id': 'seven', 'month': '1', 'day': '12'})
        assert r.status_int == 200
        assert r.body == b'variable_all: 7, day=12, id=seven, month=1'

    def test_no_remainder(self):
        try:
            r = self.app_.get('/eater')
            assert r.status_int != 200
        except Exception as ex:
            assert type(ex) == TypeError
            assert ex.args[0] in ('eater() takes exactly 2 arguments (1 given)', "eater() missing 1 required positional argument: 'id'", "TestControllerArguments.app_.<locals>.RootController.eater() missing 1 required positional argument: 'id'")

    def test_one_remainder(self):
        r = self.app_.get('/eater/1')
        assert r.status_int == 200
        assert r.body == b'eater: 1, None, '

    def test_two_remainders(self):
        r = self.app_.get('/eater/2/dummy')
        assert r.status_int == 200
        assert r.body == b'eater: 2, dummy, '

    def test_many_remainders(self):
        r = self.app_.get('/eater/3/dummy/foo/bar')
        assert r.status_int == 200
        assert r.body == b'eater: 3, dummy, foo, bar'

    def test_remainder_with_kwargs(self):
        r = self.app_.get('/eater/4?month=1&day=12')
        assert r.status_int == 200
        assert r.body == b'eater: 4, None, day=12, month=1'

    def test_remainder_with_many_kwargs(self):
        r = self.app_.get('/eater/5?id=five&month=1&day=12&dummy=dummy')
        assert r.status_int == 200
        assert r.body == b'eater: 5, dummy, day=12, month=1'

    def test_post_remainder(self):
        r = self.app_.post('/eater/6')
        assert r.status_int == 200
        assert r.body == b'eater: 6, None, '

    def test_post_three_remainders(self):
        r = self.app_.post('/eater/7/dummy')
        assert r.status_int == 200
        assert r.body == b'eater: 7, dummy, '

    def test_post_many_remainders(self):
        r = self.app_.post('/eater/8/dummy/foo/bar')
        assert r.status_int == 200
        assert r.body == b'eater: 8, dummy, foo, bar'

    def test_post_remainder_with_kwargs(self):
        r = self.app_.post('/eater/9', {'month': '1', 'day': '12'})
        assert r.status_int == 200
        assert r.body == b'eater: 9, None, day=12, month=1'

    def test_post_many_remainders_with_many_kwargs(self):
        r = self.app_.post('/eater/10', {'id': 'ten', 'month': '1', 'day': '12', 'dummy': 'dummy'})
        assert r.status_int == 200
        assert r.body == b'eater: 10, dummy, day=12, month=1'