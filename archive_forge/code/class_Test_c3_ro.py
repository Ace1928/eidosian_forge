import unittest
class Test_c3_ro(Test_ro):

    def setUp(self):
        Test_ro.setUp(self)
        from zope.testing.loggingsupport import InstalledHandler
        self.log_handler = handler = InstalledHandler('zope.interface.ro')
        self.addCleanup(handler.uninstall)

    def _callFUT(self, ob, **kwargs):
        from zope.interface.ro import ro
        return ro(ob, **kwargs)

    def _make_complex_diamond(self, base):
        O = base

        class F(O):
            pass

        class E(O):
            pass

        class D(O):
            pass

        class C(D, F):
            pass

        class B(D, E):
            pass

        class A(B, C):
            pass
        if hasattr(A, 'mro'):
            self.assertEqual(A.mro(), self._callFUT(A))
        return A

    def test_complex_diamond_object(self):
        self._make_complex_diamond(object)

    def test_complex_diamond_interface(self):
        from zope.interface import Interface
        IA = self._make_complex_diamond(Interface)
        self.assertEqual([x.__name__ for x in IA.__iro__], ['A', 'B', 'C', 'D', 'E', 'F', 'Interface'])

    def test_complex_diamond_use_legacy_argument(self):
        from zope.interface import Interface
        A = self._make_complex_diamond(Interface)
        legacy_A_iro = self._callFUT(A, use_legacy_ro=True)
        self.assertNotEqual(A.__iro__, legacy_A_iro)
        self._check_handler_complex_diamond()

    def test_complex_diamond_compare_legacy_argument(self):
        from zope.interface import Interface
        A = self._make_complex_diamond(Interface)
        computed_A_iro = self._callFUT(A, log_changed_ro=True)
        self.assertEqual(tuple(computed_A_iro), A.__iro__)
        self._check_handler_complex_diamond()

    def _check_handler_complex_diamond(self):
        handler = self.log_handler
        self.assertEqual(1, len(handler.records))
        record = handler.records[0]
        self.assertEqual('\n'.join((l.rstrip() for l in record.getMessage().splitlines())), 'Object <InterfaceClass zope.interface.tests.test_ro.A> has different legacy and C3 MROs:\n  Legacy RO (len=7)                 C3 RO (len=7; inconsistent=no)\n  ==================================================================\n    zope.interface.tests.test_ro.A    zope.interface.tests.test_ro.A\n    zope.interface.tests.test_ro.B    zope.interface.tests.test_ro.B\n  - zope.interface.tests.test_ro.E\n    zope.interface.tests.test_ro.C    zope.interface.tests.test_ro.C\n    zope.interface.tests.test_ro.D    zope.interface.tests.test_ro.D\n                                    + zope.interface.tests.test_ro.E\n    zope.interface.tests.test_ro.F    zope.interface.tests.test_ro.F\n    zope.interface.Interface          zope.interface.Interface')

    def test_ExtendedPathIndex_implement_thing_implementedby_super(self):
        from zope.interface import ro

        class _Based:
            __bases__ = ()

            def __init__(self, name, bases=(), attrs=None):
                self.__name__ = name
                self.__bases__ = bases

            def __repr__(self):
                return self.__name__
        Interface = _Based('Interface', (), {})

        class IPluggableIndex(Interface):
            pass

        class ILimitedResultIndex(IPluggableIndex):
            pass

        class IQueryIndex(IPluggableIndex):
            pass

        class IPathIndex(Interface):
            pass
        obj = _Based('object')
        PathIndex = _Based('PathIndex', (IPathIndex, IQueryIndex, obj))
        ExtendedPathIndex = _Based('ExtendedPathIndex', (ILimitedResultIndex, IQueryIndex, PathIndex))
        result = self._callFUT(ExtendedPathIndex, log_changed_ro=True, strict=False)
        self.assertEqual(result, [ExtendedPathIndex, ILimitedResultIndex, PathIndex, IPathIndex, IQueryIndex, IPluggableIndex, Interface, obj])
        record, = self.log_handler.records
        self.assertIn('used the legacy', record.getMessage())
        with self.assertRaises(ro.InconsistentResolutionOrderError):
            self._callFUT(ExtendedPathIndex, strict=True)

    def test_OSError_IOError(self):
        from zope.interface import providedBy
        from zope.interface.common import interfaces
        self.assertEqual(list(providedBy(OSError()).flattened()), [interfaces.IOSError, interfaces.IIOError, interfaces.IEnvironmentError, interfaces.IStandardError, interfaces.IException, interfaces.Interface])

    def test_non_orderable(self):
        import warnings
        from zope.interface import ro
        try:
            del ro.__warningregistry__
        except AttributeError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with C3Setting(ro.C3.WARN_BAD_IRO, True), C3Setting(ro.C3.STRICT_IRO, False):
                with self.assertRaises(ro.InconsistentResolutionOrderWarning):
                    super().test_non_orderable()
        IOErr, _ = self._make_IOErr()
        with self.assertRaises(ro.InconsistentResolutionOrderError):
            self._callFUT(IOErr, strict=True)
        with C3Setting(ro.C3.TRACK_BAD_IRO, True), C3Setting(ro.C3.STRICT_IRO, False):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._callFUT(IOErr)
            self.assertIn(IOErr, ro.C3.BAD_IROS)
        iro = self._callFUT(IOErr, strict=False)
        legacy_iro = self._callFUT(IOErr, use_legacy_ro=True, strict=False)
        self.assertEqual(iro, legacy_iro)