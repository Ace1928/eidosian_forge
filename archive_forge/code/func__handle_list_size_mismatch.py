from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Union
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER
from torch.ao.quantization.qconfig import QConfigAny
def _handle_list_size_mismatch(self, qconfig_list: List[QConfigAny], style: str) -> None:
    if len(qconfig_list) > len(self.qconfig_mappings_list):
        new_qconfig_mapping = QConfigMapping()
        for qconfig_mapping in self.qconfig_mappings_list:
            for check_style in _QCONFIG_STYLE_ORDER[1:]:
                qconfigs_dict = getattr(qconfig_mapping, check_style)
                target_qconfigs_dict = getattr(new_qconfig_mapping, check_style)
                for key in qconfigs_dict:
                    target_qconfigs_dict[key] = None
            break
        while len(qconfig_list) > len(self.qconfig_mappings_list):
            self.qconfig_mappings_list.append(copy.deepcopy(new_qconfig_mapping))
    else:
        while len(qconfig_list) < len(self.qconfig_mappings_list):
            qconfig_list.append(None)