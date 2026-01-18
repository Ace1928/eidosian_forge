import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from typing_extensions import override
from pytorch_lightning.profilers.profiler import Profiler
def _make_report(self) -> _TABLE_DATA:
    report = []
    for action, d in self.recorded_durations.items():
        d_tensor = torch.tensor(d)
        sum_d = torch.sum(d_tensor).item()
        report.append((action, sum_d / len(d), sum_d))
    report.sort(key=lambda x: x[1], reverse=True)
    return report