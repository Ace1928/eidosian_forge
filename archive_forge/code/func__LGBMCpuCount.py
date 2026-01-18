from typing import List
def _LGBMCpuCount(only_physical_cores: bool=True) -> int:
    return cpu_count()