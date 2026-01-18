import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
class TestAdaptationManagerWithABC(unittest.TestCase):
    """ Test the adaptation manager. """
    examples = traits.adaptation.tests.abc_examples

    def setUp(self):
        """ Prepares the test fixture before each test method is called. """
        self.adaptation_manager = AdaptationManager()

    def test_no_adapter_required(self):
        ex = self.examples
        plug = ex.UKPlug()
        uk_plug = self.adaptation_manager.adapt(plug, ex.UKPlug)
        self.assertIs(uk_plug, plug)
        uk_plug = self.adaptation_manager.adapt(plug, ex.UKStandard)
        self.assertIs(uk_plug, plug)

    def test_no_adapter_available(self):
        ex = self.examples
        plug = ex.UKPlug()
        eu_plug = self.adaptation_manager.adapt(plug, ex.EUPlug, None)
        self.assertEqual(eu_plug, None)
        eu_plug = self.adaptation_manager.adapt(plug, ex.EUStandard, None)
        self.assertEqual(eu_plug, None)

    def test_one_step_adaptation(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        plug = ex.UKPlug()
        eu_plug = self.adaptation_manager.adapt(plug, ex.EUStandard)
        self.assertIsNotNone(eu_plug)
        self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)
        eu_plug = self.adaptation_manager.adapt(plug, ex.EUPlug, None)
        self.assertIsNone(eu_plug)

    def test_adapter_chaining(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        self.adaptation_manager.register_factory(factory=ex.EUStandardToJapanStandard, from_protocol=ex.EUStandard, to_protocol=ex.JapanStandard)
        uk_plug = ex.UKPlug()
        japan_plug = self.adaptation_manager.adapt(uk_plug, ex.JapanStandard)
        self.assertIsNotNone(japan_plug)
        self.assertIsInstance(japan_plug, ex.EUStandardToJapanStandard)
        self.assertIs(japan_plug.adaptee.adaptee, uk_plug)

    def test_multiple_paths_unambiguous(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        self.adaptation_manager.register_factory(factory=ex.EUStandardToJapanStandard, from_protocol=ex.EUStandard, to_protocol=ex.JapanStandard)
        self.adaptation_manager.register_factory(factory=ex.JapanStandardToIraqStandard, from_protocol=ex.JapanStandard, to_protocol=ex.IraqStandard)
        self.adaptation_manager.register_factory(factory=ex.EUStandardToIraqStandard, from_protocol=ex.EUStandard, to_protocol=ex.IraqStandard)
        uk_plug = ex.UKPlug()
        iraq_plug = self.adaptation_manager.adapt(uk_plug, ex.IraqStandard)
        self.assertIsNotNone(iraq_plug)
        self.assertIsInstance(iraq_plug, ex.EUStandardToIraqStandard)
        self.assertIs(iraq_plug.adaptee.adaptee, uk_plug)

    def test_multiple_paths_ambiguous(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        self.adaptation_manager.register_factory(factory=ex.UKStandardToJapanStandard, from_protocol=ex.UKStandard, to_protocol=ex.JapanStandard)
        self.adaptation_manager.register_factory(factory=ex.JapanStandardToIraqStandard, from_protocol=ex.JapanStandard, to_protocol=ex.IraqStandard)
        self.adaptation_manager.register_factory(factory=ex.EUStandardToIraqStandard, from_protocol=ex.EUStandard, to_protocol=ex.IraqStandard)
        uk_plug = ex.UKPlug()
        iraq_plug = self.adaptation_manager.adapt(uk_plug, ex.IraqStandard)
        self.assertIsNotNone(iraq_plug)
        self.assertIn(type(iraq_plug), [ex.EUStandardToIraqStandard, ex.JapanStandardToIraqStandard])
        self.assertIs(iraq_plug.adaptee.adaptee, uk_plug)

    def test_conditional_adaptation(self):
        ex = self.examples

        def travel_plug_to_eu_standard(adaptee):
            if adaptee.mode == 'Europe':
                return ex.TravelPlugToEUStandard(adaptee=adaptee)
            else:
                return None
        self.adaptation_manager.register_factory(factory=travel_plug_to_eu_standard, from_protocol=ex.TravelPlug, to_protocol=ex.EUStandard)
        travel_plug = ex.TravelPlug(mode='Europe')
        eu_plug = self.adaptation_manager.adapt(travel_plug, ex.EUStandard)
        self.assertIsNotNone(eu_plug)
        self.assertIsInstance(eu_plug, ex.TravelPlugToEUStandard)
        travel_plug = ex.TravelPlug(mode='Asia')
        eu_plug = self.adaptation_manager.adapt(travel_plug, ex.EUStandard, None)
        self.assertIsNone(eu_plug)

    def test_spillover_adaptation_behavior(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.FileTypeToIEditor, from_protocol=ex.FileType, to_protocol=ex.IEditor)
        self.adaptation_manager.register_factory(factory=ex.IScriptableToIUndoable, from_protocol=ex.IScriptable, to_protocol=ex.IUndoable)
        file_type = ex.FileType()
        printable = self.adaptation_manager.adapt(file_type, ex.IUndoable, None)
        self.assertIsNone(printable)

    def test_adaptation_prefers_subclasses(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.TextEditorToIPrintable, from_protocol=ex.TextEditor, to_protocol=ex.IPrintable)
        self.adaptation_manager.register_factory(factory=ex.EditorToIPrintable, from_protocol=ex.Editor, to_protocol=ex.IPrintable)
        text_editor = ex.TextEditor()
        printable = self.adaptation_manager.adapt(text_editor, ex.IPrintable)
        self.assertIsNotNone(printable)
        self.assertIs(type(printable), ex.TextEditorToIPrintable)

    def test_adaptation_prefers_subclasses_other_registration_order(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.EditorToIPrintable, from_protocol=ex.Editor, to_protocol=ex.IPrintable)
        self.adaptation_manager.register_factory(factory=ex.TextEditorToIPrintable, from_protocol=ex.TextEditor, to_protocol=ex.IPrintable)
        text_editor = ex.TextEditor()
        printable = self.adaptation_manager.adapt(text_editor, ex.IPrintable)
        self.assertIsNotNone(printable)
        self.assertIs(type(printable), ex.TextEditorToIPrintable)

    def test_circular_adaptation(self):

        class Foo(object):
            pass

        class Bar(object):
            pass
        self.adaptation_manager.register_factory(factory=lambda adaptee: Foo(), from_protocol=object, to_protocol=Foo)
        self.adaptation_manager.register_factory(factory=lambda adaptee: [], from_protocol=Foo, to_protocol=object)
        obj = []
        bar = self.adaptation_manager.adapt(obj, Bar, None)
        self.assertIsNone(bar)

    def test_default_argument_in_adapt(self):
        from traits.adaptation.adaptation_manager import AdaptationError
        with self.assertRaises(AdaptationError):
            self.adaptation_manager.adapt('string', int)
        default = 'default'
        result = self.adaptation_manager.adapt('string', int, default=default)
        self.assertIs(result, default)

    def test_prefer_specific_interfaces(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.IIntermediateToITarget, from_protocol=ex.IIntermediate, to_protocol=ex.ITarget)
        self.adaptation_manager.register_factory(factory=ex.IHumanToIIntermediate, from_protocol=ex.IHuman, to_protocol=ex.IIntermediate)
        self.adaptation_manager.register_factory(factory=ex.IChildToIIntermediate, from_protocol=ex.IChild, to_protocol=ex.IIntermediate)
        self.adaptation_manager.register_factory(factory=ex.IPrimateToIIntermediate, from_protocol=ex.IPrimate, to_protocol=ex.IIntermediate)
        source = ex.Source()
        target = self.adaptation_manager.adapt(source, ex.ITarget)
        self.assertIsNotNone(target)
        self.assertIs(type(target.adaptee), ex.IChildToIIntermediate)

    def test_chaining_with_intermediate_mro_climbing(self):
        ex = self.examples
        self.adaptation_manager.register_factory(factory=ex.IStartToISpecific, from_protocol=ex.IStart, to_protocol=ex.ISpecific)
        self.adaptation_manager.register_factory(factory=ex.IGenericToIEnd, from_protocol=ex.IGeneric, to_protocol=ex.IEnd)
        start = ex.Start()
        end = self.adaptation_manager.adapt(start, ex.IEnd)
        self.assertIsNotNone(end)
        self.assertIs(type(end), ex.IGenericToIEnd)

    def test_conditional_recycling(self):

        class A(object):

            def __init__(self, allow_adaptation):
                self.allow_adaptation = allow_adaptation

        class B(object):
            pass

        class C(object):
            pass

        class D(object):
            pass
        self.adaptation_manager.register_factory(factory=lambda adaptee: A(False), from_protocol=C, to_protocol=A)
        self.adaptation_manager.register_factory(factory=lambda adaptee: A(True), from_protocol=D, to_protocol=A)
        self.adaptation_manager.register_factory(factory=lambda adaptee: D(), from_protocol=C, to_protocol=D)

        def a_to_b_adapter(adaptee):
            if adaptee.allow_adaptation:
                b = B()
                b.marker = True
            else:
                b = None
            return b
        self.adaptation_manager.register_factory(factory=a_to_b_adapter, from_protocol=A, to_protocol=B)
        c = C()
        b = self.adaptation_manager.adapt(c, B)
        self.assertIsNotNone(b)
        self.assertTrue(hasattr(b, 'marker'))

    def test_provides_protocol_for_interface_subclass(self):
        from traits.api import Interface

        class IA(Interface):
            pass

        class IB(IA):
            pass
        self.assertTrue(self.adaptation_manager.provides_protocol(IB, IA))

    def test_register_provides(self):
        from traits.api import Interface

        class IFoo(Interface):
            pass
        obj = {}
        self.assertEqual(None, self.adaptation_manager.adapt(obj, IFoo, None))
        self.adaptation_manager.register_provides(dict, IFoo)
        self.assertEqual(obj, self.adaptation_manager.adapt(obj, IFoo))