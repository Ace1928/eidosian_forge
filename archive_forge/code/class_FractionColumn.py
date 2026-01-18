from warnings import warn
from rich.progress import (
from .std import TqdmExperimentalWarning
from .std import tqdm as std_tqdm
class FractionColumn(ProgressColumn):
    """Renders completed/total, e.g. '0.5/2.3 G'."""

    def __init__(self, unit_scale=False, unit_divisor=1000):
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Calculate common unit for completed and total."""
        completed = int(task.completed)
        total = int(task.total)
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(total, ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'], self.unit_divisor)
        else:
            unit, suffix = filesize.pick_unit_and_suffix(total, [''], 1)
        precision = 0 if unit == 1 else 1
        return Text(f'{completed / unit:,.{precision}f}/{total / unit:,.{precision}f} {suffix}', style='progress.download')