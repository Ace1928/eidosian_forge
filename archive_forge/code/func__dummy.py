from __future__ import annotations
from ..runtime import driver
def _dummy(self):
    assert InfoFromBackendForTensorMap.n < InfoFromBackendForTensorMap.N
    if InfoFromBackendForTensorMap.n == 0:
        self.tensorDataType = driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_FLOAT16']
        self.tensorRank = 4
        self.globalAddressArgIdx = 0
        self.globalStridesArgIdx = [7, 6, -1, -1]
        self.globalDimsArgIdx = [5, 3, -1, -1]
        self.boxDims = [16, 64, 1, 1]
        self.elementStrides = [1, 1, 1, 1]
        self.interleave = driver.utils.CUtensorMapInterleave['CU_TENSOR_MAP_INTERLEAVE_NONE']
        self.swizzle = driver.utils.CUtensorMapSwizzle['CU_TENSOR_MAP_SWIZZLE_32B']
        self.l2Promotion = driver.utils.CUtensorMapL2promotion['CU_TENSOR_MAP_L2_PROMOTION_L2_128B']
        self.TMADescArgIdx = 11
        self.oobFill = driver.utils.CUtensorMapFloatOOBfill['CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE']
        InfoFromBackendForTensorMap.n += 1
        return
    if InfoFromBackendForTensorMap.n == 1:
        self.tensorDataType = driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_FLOAT16']
        self.tensorRank = 4
        self.globalAddressArgIdx = 1
        self.globalStridesArgIdx = [7, 6, -1, -1]
        self.globalDimsArgIdx = [5, 3, -1, -1]
        self.boxDims = [16, 64, 1, 1]
        self.elementStrides = [1, 1, 1, 1]
        self.interleave = driver.utils.CUtensorMapInterleave['CU_TENSOR_MAP_INTERLEAVE_NONE']
        self.swizzle = driver.utils.CUtensorMapSwizzle['CU_TENSOR_MAP_SWIZZLE_32B']
        self.l2Promotion = driver.utils.CUtensorMapL2promotion['CU_TENSOR_MAP_L2_PROMOTION_L2_128B']
        self.TMADescArgIdx = 12
        self.oobFill = driver.utils.CUtensorMapFloatOOBfill['CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE']
        InfoFromBackendForTensorMap.n += 1
        return