from glance_store import capabilities as caps
from glance_store.tests import base
class FakeStoreWithMixedCapabilities(caps.StoreCapability):
    _CAPABILITIES = caps.BitMasks.READ_RANDOM

    def __init__(self):
        super(FakeStoreWithMixedCapabilities, self).__init__()
        self.set_capabilities(caps.BitMasks.DRIVER_REUSABLE)