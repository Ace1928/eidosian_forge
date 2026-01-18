from glance_store import capabilities as caps
from glance_store.tests import base
class FakeStoreWithDynamicCapabilities(caps.StoreCapability):

    def __init__(self, *cap_list):
        super(FakeStoreWithDynamicCapabilities, self).__init__()
        if not cap_list:
            cap_list = [caps.BitMasks.READ_RANDOM, caps.BitMasks.DRIVER_REUSABLE]
        self.set_capabilities(*cap_list)