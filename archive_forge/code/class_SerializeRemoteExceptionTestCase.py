import sys
from oslo_serialization import jsonutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import common as exceptions
from oslo_messaging.tests import utils as test_utils
class SerializeRemoteExceptionTestCase(test_utils.BaseTestCase):
    _add_remote = [('add_remote', dict(add_remote=True)), ('do_not_add_remote', dict(add_remote=False))]
    _exception_types = [('bog_standard', dict(cls=Exception, args=['test'], kwargs={}, clsname='Exception', modname=EXCEPTIONS_MODULE, msg='test')), ('nova_style', dict(cls=NovaStyleException, args=[], kwargs={}, clsname='NovaStyleException', modname=__name__, msg='I am Nova')), ('nova_style_with_msg', dict(cls=NovaStyleException, args=['testing'], kwargs={}, clsname='NovaStyleException', modname=__name__, msg='testing')), ('kwargs_style', dict(cls=KwargsStyleException, args=[], kwargs={'who': 'Oslo'}, clsname='KwargsStyleException', modname=__name__, msg='I am Oslo'))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._add_remote, cls._exception_types)

    def test_serialize_remote_exception(self):
        try:
            try:
                raise self.cls(*self.args, **self.kwargs)
            except Exception as ex:
                cls_error = ex
                if self.add_remote:
                    ex = add_remote_postfix(ex)
                raise ex
        except Exception:
            exc_info = sys.exc_info()
        serialized = exceptions.serialize_remote_exception(exc_info)
        failure = jsonutils.loads(serialized)
        self.assertEqual(self.clsname, failure['class'], failure)
        self.assertEqual(self.modname, failure['module'])
        self.assertEqual(self.msg, failure['message'])
        self.assertEqual([self.msg], failure['args'])
        self.assertEqual(self.kwargs, failure['kwargs'])
        tb = cls_error.__class__.__name__ + ': ' + self.msg
        self.assertIn(tb, ''.join(failure['tb']))