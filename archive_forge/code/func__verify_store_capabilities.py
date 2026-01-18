from glance_store import capabilities as caps
from glance_store.tests import base
def _verify_store_capabilities(self, store):
    self.assertTrue(store.is_capable(caps.BitMasks.READ_RANDOM))
    self.assertTrue(store.is_capable(caps.BitMasks.DRIVER_REUSABLE))
    self.assertFalse(store.is_capable(caps.BitMasks.WRITE_ACCESS))