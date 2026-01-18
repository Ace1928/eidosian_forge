from __future__ import absolute_import, unicode_literals
import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase
import pytest
import six
from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp
class TextTestMixin(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def test__init__(self):
        raise NotImplementedError

    @abstractmethod
    def test__str__(self):
        raise NotImplementedError

    @abstractmethod
    def test__eq__(self):
        raise NotImplementedError

    @abstractmethod
    def test__len__(self):
        raise NotImplementedError

    @abstractmethod
    def test__contains__(self):
        raise NotImplementedError

    @abstractmethod
    def test__getitem__(self):
        raise NotImplementedError

    @abstractmethod
    def test__add__(self):
        raise NotImplementedError

    @abstractmethod
    def test_append(self):
        raise NotImplementedError

    @abstractmethod
    def test_join(self):
        raise NotImplementedError

    @abstractmethod
    def test_split(self):
        raise NotImplementedError

    @abstractmethod
    def test_startswith(self):
        raise NotImplementedError

    @abstractmethod
    def test_endswith(self):
        raise NotImplementedError

    @abstractmethod
    def test_isalpha(self):
        raise NotImplementedError

    @abstractmethod
    def test_capfirst(self):
        raise NotImplementedError

    @abstractmethod
    def test_capitalize(self):
        raise NotImplementedError

    @abstractmethod
    def test_add_period(self):
        raise NotImplementedError

    @abstractmethod
    def test_lower(self):
        raise NotImplementedError

    @abstractmethod
    def test_upper(self):
        raise NotImplementedError

    @abstractmethod
    def test_render_as(self):
        raise NotImplementedError