from glance_store import capabilities as caps
from glance_store.tests import base
class FakeStoreWithStaticCapabilities(caps.StoreCapability):
    _CAPABILITIES = caps.BitMasks.READ_RANDOM | caps.BitMasks.DRIVER_REUSABLE