import operator
import sys
import types
import unittest
import abc
import pytest
import six
class TestCustomizedMoves:

    def teardown_method(self, meth):
        try:
            del six._MovedItems.spam
        except AttributeError:
            pass
        try:
            del six.moves.__dict__['spam']
        except KeyError:
            pass

    def test_moved_attribute(self):
        attr = six.MovedAttribute('spam', 'foo', 'bar')
        if six.PY3:
            assert attr.mod == 'bar'
        else:
            assert attr.mod == 'foo'
        assert attr.attr == 'spam'
        attr = six.MovedAttribute('spam', 'foo', 'bar', 'lemma')
        assert attr.attr == 'lemma'
        attr = six.MovedAttribute('spam', 'foo', 'bar', 'lemma', 'theorm')
        if six.PY3:
            assert attr.attr == 'theorm'
        else:
            assert attr.attr == 'lemma'

    def test_moved_module(self):
        attr = six.MovedModule('spam', 'foo')
        if six.PY3:
            assert attr.mod == 'spam'
        else:
            assert attr.mod == 'foo'
        attr = six.MovedModule('spam', 'foo', 'bar')
        if six.PY3:
            assert attr.mod == 'bar'
        else:
            assert attr.mod == 'foo'

    def test_custom_move_module(self):
        attr = six.MovedModule('spam', 'six', 'six')
        six.add_move(attr)
        six.remove_move('spam')
        assert not hasattr(six.moves, 'spam')
        attr = six.MovedModule('spam', 'six', 'six')
        six.add_move(attr)
        from six.moves import spam
        assert spam is six
        six.remove_move('spam')
        assert not hasattr(six.moves, 'spam')

    def test_custom_move_attribute(self):
        attr = six.MovedAttribute('spam', 'six', 'six', 'u', 'u')
        six.add_move(attr)
        six.remove_move('spam')
        assert not hasattr(six.moves, 'spam')
        attr = six.MovedAttribute('spam', 'six', 'six', 'u', 'u')
        six.add_move(attr)
        from six.moves import spam
        assert spam is six.u
        six.remove_move('spam')
        assert not hasattr(six.moves, 'spam')

    def test_empty_remove(self):
        pytest.raises(AttributeError, six.remove_move, 'eggs')