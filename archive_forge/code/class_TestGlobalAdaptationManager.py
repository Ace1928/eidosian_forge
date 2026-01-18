import unittest
from traits.adaptation.api import (
import traits.adaptation.tests.abc_examples
class TestGlobalAdaptationManager(unittest.TestCase):
    """ Test the setting/getting/resetting/using the global adaptation manager.
    """
    examples = traits.adaptation.tests.abc_examples

    def setUp(self):
        """ Prepares the test fixture before each test method is called. """
        reset_global_adaptation_manager()

    def test_reset_adaptation_manager(self):
        ex = self.examples
        adaptation_manager = get_global_adaptation_manager()
        adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        uk_plug = ex.UKPlug()
        reset_global_adaptation_manager()
        adaptation_manager = get_global_adaptation_manager()
        with self.assertRaises(AdaptationError):
            adaptation_manager.adapt(uk_plug, ex.EUStandard)

    def test_set_adaptation_manager(self):
        ex = self.examples
        adaptation_manager = AdaptationManager()
        adaptation_manager.register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        uk_plug = ex.UKPlug()
        set_global_adaptation_manager(adaptation_manager)
        global_adaptation_manager = get_global_adaptation_manager()
        eu_plug = global_adaptation_manager.adapt(uk_plug, ex.EUStandard)
        self.assertIsNotNone(eu_plug)
        self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)

    def test_global_convenience_functions(self):
        ex = self.examples
        register_factory(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        uk_plug = ex.UKPlug()
        eu_plug = adapt(uk_plug, ex.EUStandard)
        self.assertIsNotNone(eu_plug)
        self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)
        self.assertTrue(provides_protocol(ex.UKPlug, ex.UKStandard))
        self.assertTrue(supports_protocol(uk_plug, ex.EUStandard))

    def test_global_register_provides(self):
        from traits.api import Interface

        class IFoo(Interface):
            pass
        obj = {}
        register_provides(dict, IFoo)
        self.assertEqual(obj, adapt(obj, IFoo))

    def test_global_register_offer(self):
        ex = self.examples
        offer = AdaptationOffer(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
        register_offer(offer)
        uk_plug = ex.UKPlug()
        eu_plug = adapt(uk_plug, ex.EUStandard)
        self.assertIsNotNone(eu_plug)
        self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)