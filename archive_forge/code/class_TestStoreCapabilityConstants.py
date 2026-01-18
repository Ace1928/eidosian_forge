from glance_store import capabilities as caps
from glance_store.tests import base
class TestStoreCapabilityConstants(base.StoreBaseTest):

    def test_one_single_capability_own_one_bit(self):
        cap_list = [caps.BitMasks.READ_ACCESS, caps.BitMasks.WRITE_ACCESS, caps.BitMasks.DRIVER_REUSABLE]
        for cap in cap_list:
            self.assertEqual(1, bin(cap).count('1'))

    def test_combined_capability_bits(self):
        check = caps.StoreCapability.contains
        check(caps.BitMasks.READ_OFFSET, caps.BitMasks.READ_ACCESS)
        check(caps.BitMasks.READ_CHUNK, caps.BitMasks.READ_ACCESS)
        check(caps.BitMasks.READ_RANDOM, caps.BitMasks.READ_CHUNK)
        check(caps.BitMasks.READ_RANDOM, caps.BitMasks.READ_OFFSET)
        check(caps.BitMasks.WRITE_OFFSET, caps.BitMasks.WRITE_ACCESS)
        check(caps.BitMasks.WRITE_CHUNK, caps.BitMasks.WRITE_ACCESS)
        check(caps.BitMasks.WRITE_RANDOM, caps.BitMasks.WRITE_CHUNK)
        check(caps.BitMasks.WRITE_RANDOM, caps.BitMasks.WRITE_OFFSET)
        check(caps.BitMasks.RW_ACCESS, caps.BitMasks.READ_ACCESS)
        check(caps.BitMasks.RW_ACCESS, caps.BitMasks.WRITE_ACCESS)
        check(caps.BitMasks.RW_OFFSET, caps.BitMasks.READ_OFFSET)
        check(caps.BitMasks.RW_OFFSET, caps.BitMasks.WRITE_OFFSET)
        check(caps.BitMasks.RW_CHUNK, caps.BitMasks.READ_CHUNK)
        check(caps.BitMasks.RW_CHUNK, caps.BitMasks.WRITE_CHUNK)
        check(caps.BitMasks.RW_RANDOM, caps.BitMasks.READ_RANDOM)
        check(caps.BitMasks.RW_RANDOM, caps.BitMasks.WRITE_RANDOM)