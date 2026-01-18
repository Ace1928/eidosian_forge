from zope.interface._compat import _should_attempt_c_optimizations
class MissingSomeAttrs:
    """
    Helper for tests that raises a specific exception
    for attributes that are missing. This is usually not
    an AttributeError, and this object is used to test that
    those errors are not improperly caught and treated like
    an AttributeError.
    """

    def __init__(self, exc_kind, **other_attrs):
        self.__exc_kind = exc_kind
        d = object.__getattribute__(self, '__dict__')
        d.update(other_attrs)

    def __getattribute__(self, name):
        d = object.__getattribute__(self, '__dict__')
        try:
            return d[name]
        except KeyError:
            raise d['_MissingSomeAttrs__exc_kind'](name)
    EXCEPTION_CLASSES = (TypeError, RuntimeError, BaseException, ValueError)

    @classmethod
    def test_raises(cls, unittest, test_func, expected_missing, **other_attrs):
        """
        Loop through various exceptions, calling *test_func* inside a ``assertRaises`` block.

        :param test_func: A callable of one argument, the instance of this
            class.
        :param str expected_missing: The attribute that should fail with the exception.
           This is used to ensure that we're testing the path we think we are.
        :param other_attrs: Attributes that should be provided on the test object.
           Must not contain *expected_missing*.
        """
        assert isinstance(expected_missing, str)
        assert expected_missing not in other_attrs
        for exc in cls.EXCEPTION_CLASSES:
            ob = cls(exc, **other_attrs)
            with unittest.assertRaises(exc) as ex:
                test_func(ob)
            unittest.assertEqual(ex.exception.args[0], expected_missing)
        ob = cls(AttributeError, **other_attrs)
        try:
            test_func(ob)
        except AttributeError as e:
            unittest.assertNotIn(expected_missing, str(e))
        except Exception:
            pass