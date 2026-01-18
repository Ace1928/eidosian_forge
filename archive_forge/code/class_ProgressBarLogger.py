from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
class ProgressBarLogger(ProgressLogger):
    """Generic class for progress loggers.

    A progress logger contains a "state" dictionnary

    Parameters
    ----------

    init_state
      Initial state of the logger

    bars
      Either None (will be initialized with no bar) or a list/tuple of bar
      names (``['main', 'sub']``) which will be initialized with index -1 and
      no total, or a dictionary (possibly ordered) of bars, of the form
      ``{bar_1: {title: 'bar1', index: 2, total:23}, bar_2: {...}}``

    ignored_bars
      Either None (newly met bars will be added) or a list of blacklisted bar
      names, or ``'all_others'`` to signify that all bar names not already in
      ``self.bars`` will be ignored.
    """
    bar_indent = 2

    def __init__(self, init_state=None, bars=None, ignored_bars=None, logged_bars='all', min_time_interval=0, ignore_bars_under=0):
        ProgressLogger.__init__(self, init_state)
        if bars is None:
            bars = OrderedDict()
        elif isinstance(bars, (list, tuple)):
            bars = OrderedDict([(b, dict(title=b, index=-1, total=None, message=None, indent=0)) for b in bars])
        if isinstance(ignored_bars, (list, tuple)):
            ignored_bars = set(ignored_bars)
        self.ignored_bars = ignored_bars
        self.logged_bars = logged_bars
        self.state['bars'] = bars
        self.min_time_interval = min_time_interval
        self.ignore_bars_under = ignore_bars_under

    @property
    def bars(self):
        """Return ``self.state['bars'].``"""
        return self.state['bars']

    def bar_is_ignored(self, bar):
        if self.ignored_bars is None:
            return False
        elif self.ignored_bars == 'all_others':
            return bar not in self.bars
        else:
            return bar in self.ignored_bars

    def bar_is_logged(self, bar):
        if not self.logged_bars:
            return False
        elif self.logged_bars == 'all':
            return True
        else:
            return bar in self.logged_bars

    def iterable_is_too_short(self, iterable):
        length = len(iterable) if hasattr(iterable, '__len__') else None
        return length is not None and length < self.ignore_bars_under

    def iter_bar(self, bar_prefix='', **kw):
        """Iterate through a list while updating a state bar.

        Examples
        --------
        >>> for username in logger.iter_bar(user=['tom', 'tim', 'lea']):
        >>>     # At every loop, logger.state['bars']['user'] is updated
        >>>     # to {index: i, total: 3, title:'user'}
        >>>     print (username)

        """
        if 'bar_message' in kw:
            bar_message = kw.pop('bar_message')
        else:
            bar_message = None
        bar, iterable = kw.popitem()
        if self.bar_is_ignored(bar) or self.iterable_is_too_short(iterable):
            return iterable
        bar = bar_prefix + bar
        if hasattr(iterable, '__len__'):
            self(**{bar + '__total': len(iterable)})

        def new_iterable():
            last_time = time.time()
            i = 0
            for i, it in enumerate(iterable):
                now_time = time.time()
                if i == 0 or now_time - last_time > self.min_time_interval:
                    if bar_message is not None:
                        self(**{bar + '__message': bar_message(it)})
                    self(**{bar + '__index': i})
                    last_time = now_time
                yield it
            if self.bars[bar]['index'] != i:
                self(**{bar + '__index': i})
            self(**{bar + '__index': i + 1})
        return new_iterable()

    def bars_callback(self, bar, attr, value, old_value=None):
        """Execute a custom action after the progress bars are updated.

        Parameters
        ----------
        bar
          Name/ID of the bar to be modified.

        attr
          Attribute of the bar attribute to be modified

        value
          New value of the attribute

        old_value
          Previous value of this bar's attribute.

        This default callback does nothing, overwrite it by subclassing.
        """
        pass

    def __call__(self, **kw):
        items = sorted(kw.items(), key=lambda kv: not kv[0].endswith('total'))
        for key, value in items:
            if '__' in key:
                bar, attr = key.split('__')
                if self.bar_is_ignored(bar):
                    continue
                kw.pop(key)
                if bar not in self.bars:
                    self.bars[bar] = dict(title=bar, index=-1, total=None, message=None)
                old_value = self.bars[bar][attr]
                if self.bar_is_logged(bar):
                    new_bar = attr == 'index' and value < old_value
                    if attr == 'total' or new_bar:
                        self.bars[bar]['indent'] = self.log_indent
                    else:
                        self.log_indent = self.bars[bar]['indent']
                    self.log('[%s] %s: %s' % (bar, attr, value))
                    self.log_indent += self.bar_indent
                self.bars[bar][attr] = value
                self.bars_callback(bar, attr, value, old_value)
        self.state.update(kw)
        self.callback(**kw)