from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def _match_c_sig(self, c_sig: str):
    m = self.c_sig.findall(c_sig)
    if len(m):
        tys, args = ([], [])
        for ty, arg_name in m:
            tys.append(ty)
            args.append(arg_name)
        return (tys, args)
    raise LinkerError(f'{c_sig} is not a valid argument signature')