from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
class Test_callback_registry:

    def setup_method(self):
        self.signal = 'test'
        self.callbacks = cbook.CallbackRegistry()

    def connect(self, s, func, pickle):
        if pickle:
            return self.callbacks.connect(s, func)
        else:
            return self.callbacks._connect_picklable(s, func)

    def disconnect(self, cid):
        return self.callbacks.disconnect(cid)

    def count(self):
        count1 = len(self.callbacks._func_cid_map.get(self.signal, []))
        count2 = len(self.callbacks.callbacks.get(self.signal))
        assert count1 == count2
        return count1

    def is_empty(self):
        np.testing.break_cycles()
        assert self.callbacks._func_cid_map == {}
        assert self.callbacks.callbacks == {}
        assert self.callbacks._pickled_cids == set()

    def is_not_empty(self):
        np.testing.break_cycles()
        assert self.callbacks._func_cid_map != {}
        assert self.callbacks.callbacks != {}

    def test_cid_restore(self):
        cb = cbook.CallbackRegistry()
        cb.connect('a', lambda: None)
        cb2 = pickle.loads(pickle.dumps(cb))
        cid = cb2.connect('c', lambda: None)
        assert cid == 1

    @pytest.mark.parametrize('pickle', [True, False])
    def test_callback_complete(self, pickle):
        self.is_empty()
        mini_me = Test_callback_registry()
        cid1 = self.connect(self.signal, mini_me.dummy, pickle)
        assert type(cid1) is int
        self.is_not_empty()
        cid2 = self.connect(self.signal, mini_me.dummy, pickle)
        assert cid1 == cid2
        self.is_not_empty()
        assert len(self.callbacks._func_cid_map) == 1
        assert len(self.callbacks.callbacks) == 1
        del mini_me
        self.is_empty()

    @pytest.mark.parametrize('pickle', [True, False])
    def test_callback_disconnect(self, pickle):
        self.is_empty()
        mini_me = Test_callback_registry()
        cid1 = self.connect(self.signal, mini_me.dummy, pickle)
        assert type(cid1) is int
        self.is_not_empty()
        self.disconnect(cid1)
        self.is_empty()

    @pytest.mark.parametrize('pickle', [True, False])
    def test_callback_wrong_disconnect(self, pickle):
        self.is_empty()
        mini_me = Test_callback_registry()
        cid1 = self.connect(self.signal, mini_me.dummy, pickle)
        assert type(cid1) is int
        self.is_not_empty()
        self.disconnect('foo')
        self.is_not_empty()

    @pytest.mark.parametrize('pickle', [True, False])
    def test_registration_on_non_empty_registry(self, pickle):
        self.is_empty()
        mini_me = Test_callback_registry()
        self.connect(self.signal, mini_me.dummy, pickle)
        mini_me2 = Test_callback_registry()
        self.connect(self.signal, mini_me2.dummy, pickle)
        mini_me2 = Test_callback_registry()
        self.connect(self.signal, mini_me2.dummy, pickle)
        self.is_not_empty()
        assert self.count() == 2
        mini_me = None
        mini_me2 = None
        self.is_empty()

    def dummy(self):
        pass

    def test_pickling(self):
        assert hasattr(pickle.loads(pickle.dumps(cbook.CallbackRegistry())), 'callbacks')