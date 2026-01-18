import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def _find_match(str_list: Union[Dict[str, Any], List[str]], key_str: str, postfix: str) -> Optional[str]:
    split_str = key_str.split('.')
    if split_str[-1] == postfix:
        match_string = ''.join(key_str.split('.')[0:-1])
        for s2 in str_list:
            pattern1 = ''.join(s2.split('.')[0:-1])
            pattern2 = ''.join(s2.split('.')[0:-2])
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2
        if postfix == '_packed_params':
            match_string = ''.join(key_str.split('.')[0:-2])
            if len(match_string) == 0:
                return None
            for s2 in str_list:
                pattern1 = ''.join(s2.split('.')[0:-1])
                pattern2 = ''.join(s2.split('.')[0:-2])
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
        return None
    else:
        return None