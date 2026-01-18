from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class VersionTests(TestCase):

    def test_nullVersionUpgrade(self) -> None:
        global NullVersioned

        class NullVersioned:

            def __init__(self) -> None:
                self.ok = 0
        pkcl = pickle.dumps(NullVersioned())

        class NullVersioned(styles.Versioned):
            persistenceVersion = 1

            def upgradeToVersion1(self) -> None:
                self.ok = 1
        mnv = pickle.loads(pkcl)
        styles.doUpgrade()
        assert mnv.ok, 'initial upgrade not run!'

    def test_versionUpgrade(self) -> None:
        global MyVersioned

        class MyVersioned(styles.Versioned):
            persistenceVersion = 2
            persistenceForgets = ['garbagedata']
            v3 = 0
            v4 = 0

            def __init__(self) -> None:
                self.somedata = 'xxx'
                self.garbagedata = lambda q: 'cant persist'

            def upgradeToVersion3(self) -> None:
                self.v3 += 1

            def upgradeToVersion4(self) -> None:
                self.v4 += 1
        mv = MyVersioned()
        assert not (mv.v3 or mv.v4), "hasn't been upgraded yet"
        pickl = pickle.dumps(mv)
        MyVersioned.persistenceVersion = 4
        obj = pickle.loads(pickl)
        styles.doUpgrade()
        assert obj.v3, "didn't do version 3 upgrade"
        assert obj.v4, "didn't do version 4 upgrade"
        pickl = pickle.dumps(obj)
        obj = pickle.loads(pickl)
        styles.doUpgrade()
        assert obj.v3 == 1, 'upgraded unnecessarily'
        assert obj.v4 == 1, 'upgraded unnecessarily'

    def test_nonIdentityHash(self) -> None:
        global ClassWithCustomHash

        class ClassWithCustomHash(styles.Versioned):

            def __init__(self, unique: str, hash: int) -> None:
                self.unique = unique
                self.hash = hash

            def __hash__(self) -> int:
                return self.hash
        v1 = ClassWithCustomHash('v1', 0)
        v2 = ClassWithCustomHash('v2', 0)
        pkl = pickle.dumps((v1, v2))
        del v1, v2
        ClassWithCustomHash.persistenceVersion = 1
        ClassWithCustomHash.upgradeToVersion1 = lambda self: setattr(self, 'upgraded', True)
        v1, v2 = pickle.loads(pkl)
        styles.doUpgrade()
        self.assertEqual(v1.unique, 'v1')
        self.assertEqual(v2.unique, 'v2')
        self.assertTrue(v1.upgraded)
        self.assertTrue(v2.upgraded)

    def test_upgradeDeserializesObjectsRequiringUpgrade(self) -> None:
        global ToyClassA, ToyClassB

        class ToyClassA(styles.Versioned):
            pass

        class ToyClassB(styles.Versioned):
            pass
        x = ToyClassA()
        y = ToyClassB()
        pklA, pklB = (pickle.dumps(x), pickle.dumps(y))
        del x, y
        ToyClassA.persistenceVersion = 1

        def upgradeToVersion1(self: Any) -> None:
            self.y = pickle.loads(pklB)
            styles.doUpgrade()
        ToyClassA.upgradeToVersion1 = upgradeToVersion1
        ToyClassB.persistenceVersion = 1

        def setUpgraded(self: object) -> None:
            setattr(self, 'upgraded', True)
        ToyClassB.upgradeToVersion1 = setUpgraded
        x = pickle.loads(pklA)
        styles.doUpgrade()
        self.assertTrue(x.y.upgraded)