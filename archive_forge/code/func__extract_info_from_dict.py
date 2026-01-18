from __future__ import annotations
from ..runtime import driver
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