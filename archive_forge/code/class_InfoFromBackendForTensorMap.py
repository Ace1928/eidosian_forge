from __future__ import annotations
from ..runtime import driver
class InfoFromBackendForTensorMap:
    N = 2
    n = 0
    ntma = 0

    def __init__(self, infos=None, dummy=False):
        self.dummy = dummy
        self.ids_of_folded_args = ()
        if not dummy and (not isinstance(infos, dict)):
            self._extract_info_from_backend(infos)
        elif not dummy and isinstance(infos, dict):
            self._extract_info_from_dict(infos)
        elif dummy:
            self._dummy()

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

    def _extract_info_from_backend(self, infos):
        self.tensorDataType = infos.tensorDataType
        self.tensorRank = infos.tensorRank
        self.globalAddressArgIdx = infos.globalAddressArgIdx
        self.globalStridesArgIdx = infos.globalStridesArgIdx
        self.globalDimsArgIdx = infos.globalDimsArgIdx
        self.boxDims = infos.boxDims
        self.elementStrides = infos.elementStrides
        self.interleave = infos.interleave
        self.swizzle = infos.swizzle
        self.l2Promotion = infos.l2Promotion
        self.oobFill = infos.oobFill
        self.TMADescArgIdx = infos.TMADescArgIdx

    def _extract_info_from_dict(self, infos: dict):
        self.tensorDataType = infos['tensorDataType']
        self.tensorRank = infos['tensorRank']
        self.globalAddressArgIdx = infos['globalAddressArgIdx']
        self.globalStridesArgIdx = infos['globalStridesArgIdx']
        self.globalDimsArgIdx = infos['globalDimsArgIdx']
        self.boxDims = infos['boxDims']
        self.elementStrides = infos['elementStrides']
        self.interleave = infos['interleave']
        self.swizzle = infos['swizzle']
        self.l2Promotion = infos['l2Promotion']
        self.oobFill = infos['oobFill']
        self.TMADescArgIdx = infos['TMADescArgIdx']

    def get_address_tma_mapping(self):
        return {self.globalAddressArgIdx: self.TMADescArgIdx + len(self.ids_of_folded_args)}

    def get_id_of_tensormap(self):
        return self.TMADescArgIdx + len(self.ids_of_folded_args)

    def getTMADescArgIdx(self):
        return self.TMADescArgIdx

    def bytes_from_type(self, dtype):
        return {driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_UINT8']: 1, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_UINT16']: 2, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_UINT32']: 4, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_INT32']: 4, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_UINT64']: 8, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_INT64']: 8, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_FLOAT16']: 2, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_FLOAT32']: 4, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_FLOAT64']: 8, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_BFLOAT16']: 2, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ']: 4, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_TFLOAT32']: 4, driver.utils.CUtensorMapDataType['CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ']: 4}[dtype]

    def getTensorMapDataType(self):
        return self.tensorDataType

    def getInterleave(self):
        return self.interleave

    def getSwizzle(self):
        return self.swizzle

    def getL2Promotion(self):
        return self.l2Promotion

    def getOobFill(self):
        return self.oobFill

    def getTensorRank(self):
        return self.tensorRank

    def getBoxDims(self):
        return self.boxDims

    def getElementStrides(self):
        return self.elementStrides

    def getGlobalAddress(self, args):
        idx = self.getOriginArgIdx(self.globalAddressArgIdx, args)
        return args[idx]

    def getGlobalDims(self, args):
        shape = []
        for e in self.globalDimsArgIdx:
            t = 1
            if e == -1:
                t = 1
            elif e < 0 and e != -1:
                t = -e - 1
            else:
                idx = self.getOriginArgIdx(e, args)
                t = args[idx]
            shape.append(t)
        return shape

    def getGlobalStrides(self, args):
        t_globalDims = [int(e) for e in self.getGlobalDims(args)]
        t_globalStridesArgIdx = self.globalStridesArgIdx.copy()
        strides_in_elements = []
        for i in range(self.tensorRank):
            t = 1
            if t_globalStridesArgIdx[i] == -1:
                for ii in range(i):
                    t *= t_globalDims[ii]
            elif t_globalStridesArgIdx[i] < 0:
                t = -1 - t_globalStridesArgIdx[i]
            else:
                new_idx = self.getOriginArgIdx(t_globalStridesArgIdx[i], args)
                t = args[new_idx]
            strides_in_elements.append(t)
        strides_in_elements = strides_in_elements[1:]
        strides_in_bytes = [e * self.bytes_from_type(self.tensorDataType) for e in strides_in_elements]
        return strides_in_bytes

    def getOriginArgIdx(self, idx, args):
        if self.ids_of_folded_args:
            ids_before_folding_arg = [i for i in range(len(args)) if i not in self.ids_of_folded_args]
            return ids_before_folding_arg[idx]
        else:
            return idx

    def tensormap(self, args):
        return driver.utils.cuTensorMapEncodeTiled(self.getTensorMapDataType(), self.getTensorRank(), self.getGlobalAddress(args), self.getGlobalDims(args), self.getGlobalStrides(args), self.getBoxDims(), self.getElementStrides(), self.getInterleave(), self.getSwizzle(), self.getL2Promotion(), self.getOobFill())

    def __hash__(self):
        return hash((self.ids_of_folded_args, self.globalAddressArgIdx, tuple(self.globalDimsArgIdx), tuple(self.globalStridesArgIdx), self.tensorDataType, self.tensorRank, tuple(self.boxDims), tuple(self.elementStrides), self.interleave, self.swizzle, self.l2Promotion, self.oobFill))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.ids_of_folded_args, self.globalAddressArgIdx, self.globalDimsArgIdx, self.globalStridesArgIdx, self.tensorDataType, self.tensorRank, self.boxDims, self.elementStrides, self.interleave, self.swizzle, self.l2Promotion, self.oobFill) == (other.ids_of_folded_args, other.globalAddressArgIdx, other.globalDimsArgIdx, other.globalStridesArgIdx, other.tensorDataType, other.tensorRank, other.boxDims, other.elementStrides, other.interleave, other.swizzle, other.l2Promotion, other.oobFill)