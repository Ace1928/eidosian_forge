from warnings import warn
from rich.progress import (
from .std import TqdmExperimentalWarning
from .std import tqdm as std_tqdm
class tqdm_rich(std_tqdm):
    """Experimental rich.progress GUI version of tqdm!"""

    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        progress  : tuple, optional
            arguments for `rich.progress.Progress()`.
        options  : dict, optional
            keyword arguments for `rich.progress.Progress()`.
        """
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        kwargs['disable'] = bool(kwargs.get('disable', False))
        progress = kwargs.pop('progress', None)
        options = kwargs.pop('options', {}).copy()
        super(tqdm_rich, self).__init__(*args, **kwargs)
        if self.disable:
            return
        warn('rich is experimental/alpha', TqdmExperimentalWarning, stacklevel=2)
        d = self.format_dict
        if progress is None:
            progress = ('[progress.description]{task.description}[progress.percentage]{task.percentage:>4.0f}%', BarColumn(bar_width=None), FractionColumn(unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']), '[', TimeElapsedColumn(), '<', TimeRemainingColumn(), ',', RateColumn(unit=d['unit'], unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']), ']')
        options.setdefault('transient', not self.leave)
        self._prog = Progress(*progress, **options)
        self._prog.__enter__()
        self._task_id = self._prog.add_task(self.desc or '', **d)

    def close(self):
        if self.disable:
            return
        super(tqdm_rich, self).close()
        self._prog.__exit__(None, None, None)

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        if not hasattr(self, '_prog'):
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc)

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_prog'):
            self._prog.reset(total=total)
        super(tqdm_rich, self).reset(total=total)