from __future__ import annotations
import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
import numpy as np
import pandas as pd
import pytest
import torch
from bson.objectid import ObjectId
from monty.json import MontyDecoder, MontyEncoder, MSONable, _load_redirect, jsanitize
from . import __version__ as tests_version
class TestMSONable:

    def setup_method(self):
        self.good_cls = GoodMSONClass

        class BadMSONClass(MSONable):

            def __init__(self, a, b):
                self.a = a
                self.b = b

            def as_dict(self):
                d = {'init': {'a': self.a, 'b': self.b}}
                return d
        self.bad_cls = BadMSONClass

        class BadMSONClass2(MSONable):

            def __init__(self, a, b):
                self.a = a
                self.c = b
        self.bad_cls2 = BadMSONClass2

        class AutoMSON(MSONable):

            def __init__(self, a, b):
                self.a = a
                self.b = b
        self.auto_mson = AutoMSON

    def test_to_from_dict(self):
        obj = self.good_cls('Hello', 'World', 'Python')
        d = obj.as_dict()
        assert d is not None
        self.good_cls.from_dict(d)
        jsonstr = obj.to_json()
        d = json.loads(jsonstr)
        assert d['@class'], 'GoodMSONClass'
        obj = self.bad_cls('Hello', 'World')
        d = obj.as_dict()
        assert d is not None
        with pytest.raises(TypeError):
            self.bad_cls.from_dict(d)
        obj = self.bad_cls2('Hello', 'World')
        with pytest.raises(NotImplementedError):
            obj.as_dict()
        obj = self.auto_mson(2, 3)
        d = obj.as_dict()
        self.auto_mson.from_dict(d)

    def test_unsafe_hash(self):
        GMC = GoodMSONClass
        a_list = [GMC(1, 1.0, 'one'), GMC(2, 2.0, 'two')]
        b_dict = {'first': GMC(3, 3.0, 'three'), 'second': GMC(4, 4.0, 'four')}
        c_list_dict_list = [{'list1': [GMC(5, 5.0, 'five'), GMC(6, 6.0, 'six'), GMC(7, 7.0, 'seven')], 'list2': [GMC(8, 8.0, 'eight')]}, {'list3': [GMC(9, 9.0, 'nine'), GMC(10, 10.0, 'ten'), GMC(11, 11.0, 'eleven'), GMC(12, 12.0, 'twelve')], 'list4': [GMC(13, 13.0, 'thirteen'), GMC(14, 14.0, 'fourteen')], 'list5': [GMC(15, 15.0, 'fifteen')]}]
        obj = GoodNestedMSONClass(a_list=a_list, b_dict=b_dict, c_list_dict_list=c_list_dict_list)
        assert a_list[0].unsafe_hash().hexdigest() == 'ea44de0e2ef627be582282c02c48e94de0d58ec6'
        assert obj.unsafe_hash().hexdigest() == '44204c8da394e878f7562c9aa2e37c2177f28b81'

    def test_version(self):
        obj = self.good_cls('Hello', 'World', 'Python')
        d = obj.as_dict()
        assert d['@version'] == tests_version

    def test_nested_to_from_dict(self):
        GMC = GoodMSONClass
        a_list = [GMC(1, 1.0, 'one'), GMC(2, 2.0, 'two')]
        b_dict = {'first': GMC(3, 3.0, 'three'), 'second': GMC(4, 4.0, 'four')}
        c_list_dict_list = [{'list1': [GMC(5, 5.0, 'five'), GMC(6, 6.0, 'six'), GMC(7, 7.0, 'seven')], 'list2': [GMC(8, 8.0, 'eight')]}, {'list3': [GMC(9, 9.0, 'nine'), GMC(10, 10.0, 'ten'), GMC(11, 11.0, 'eleven'), GMC(12, 12.0, 'twelve')], 'list4': [GMC(13, 13.0, 'thirteen'), GMC(14, 14.0, 'fourteen')], 'list5': [GMC(15, 15.0, 'fifteen')]}]
        obj = GoodNestedMSONClass(a_list=a_list, b_dict=b_dict, c_list_dict_list=c_list_dict_list)
        obj_dict = obj.as_dict()
        obj2 = GoodNestedMSONClass.from_dict(obj_dict)
        assert [obj2.a_list[ii] == aa for ii, aa in enumerate(obj.a_list)]
        assert [obj2.b_dict[kk] == val for kk, val in obj.b_dict.items()]
        assert len(obj.a_list) == len(obj2.a_list)
        assert len(obj.b_dict) == len(obj2.b_dict)
        s = json.dumps(obj_dict)
        obj3 = json.loads(s, cls=MontyDecoder)
        assert [obj2.a_list[ii] == aa for ii, aa in enumerate(obj3.a_list)]
        assert [obj2.b_dict[kk] == val for kk, val in obj3.b_dict.items()]
        assert len(obj3.a_list) == len(obj2.a_list)
        assert len(obj3.b_dict) == len(obj2.b_dict)
        s = json.dumps(obj, cls=MontyEncoder)
        obj4 = json.loads(s, cls=MontyDecoder)
        assert [obj4.a_list[ii] == aa for ii, aa in enumerate(obj.a_list)]
        assert [obj4.b_dict[kk] == val for kk, val in obj.b_dict.items()]
        assert len(obj.a_list) == len(obj4.a_list)
        assert len(obj.b_dict) == len(obj4.b_dict)

    def test_enum_serialization(self):
        e = EnumTest.a
        d = e.as_dict()
        e_new = EnumTest.from_dict(d)
        assert e_new.name == e.name
        assert e_new.value == e.value
        d = {'123': EnumTest.a}
        f = jsanitize(d)
        assert f['123'] == 'EnumTest.a'
        f = jsanitize(d, strict=True)
        assert f['123']['@module'] == 'tests.test_json'
        assert f['123']['@class'] == 'EnumTest'
        assert f['123']['value'] == 1
        f = jsanitize(d, strict=True, enum_values=True)
        assert f['123'] == 1
        f = jsanitize(d, enum_values=True)
        assert f['123'] == 1