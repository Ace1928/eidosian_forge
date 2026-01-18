import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
class HierarchicalTimer(object):
    """A class for collecting and displaying hierarchical timing
    information

    When implementing an iterative algorithm with nested subroutines
    (e.g. an optimization solver), we often want to know the cumulative
    time spent in each subroutine as well as this time as a proportion
    of time spent in the calling routine. This class collects timing
    information, for user-specified keys, that accumulates over the life
    of the timer object and preserves the hierarchical (nested) structure
    of timing categories.

    Examples
    --------
    >>> import time
    >>> from pyomo.common.timing import HierarchicalTimer
    >>> timer = HierarchicalTimer()
    >>> timer.start('all')
    >>> time.sleep(0.2)
    >>> for i in range(10):
    ...     timer.start('a')
    ...     time.sleep(0.1)
    ...     for i in range(5):
    ...         timer.start('aa')
    ...         time.sleep(0.01)
    ...         timer.stop('aa')
    ...     timer.start('ab')
    ...     timer.stop('ab')
    ...     timer.stop('a')
    ...
    >>> for i in range(10):
    ...     timer.start('b')
    ...     time.sleep(0.02)
    ...     timer.stop('b')
    ...
    >>> timer.stop('all')
    >>> print(timer)       # doctest: +SKIP
    Identifier        ncalls   cumtime   percall      %
    ---------------------------------------------------
    all                    1     2.248     2.248  100.0
         ----------------------------------------------
         a                10     1.787     0.179   79.5
              -----------------------------------------
              aa          50     0.733     0.015   41.0
              ab          10     0.000     0.000    0.0
              other      n/a     1.055       n/a   59.0
              =========================================
         b                10     0.248     0.025   11.0
         other           n/a     0.213       n/a    9.5
         ==============================================
    ===================================================
    <BLANKLINE>

    The columns are:

      ncalls
          The number of times the timer was started and stopped
      cumtime
          The cumulative time (in seconds) the timer was active
          (started but not stopped)
      percall
          cumtime (in seconds) / ncalls
      "%"
          This is cumtime of the timer divided by cumtime of the
          parent timer times 100

    >>> print('a total time: %f' % timer.get_total_time('all.a'))         # doctest: +SKIP
    a total time: 1.902037
    >>> print('ab num calls: %d' % timer.get_num_calls('all.a.ab'))         # doctest: +SKIP
    ab num calls: 10
    >>> print('aa %% time: %f' % timer.get_relative_percent_time('all.a.aa'))         # doctest: +SKIP
    aa % time: 44.144148
    >>> print('aa %% total: %f' % timer.get_total_percent_time('all.a.aa'))         # doctest: +SKIP
    aa % total: 35.976058

    When implementing an algorithm, it is often useful to collect detailed
    hierarchical timing information. However, when communicating a timing
    profile, it is often best to retain only the most relevant information
    in a flattened data structure. In the following example, suppose we
    want to compare the time spent in the ``"c"`` and ``"f"`` subroutines.
    We would like to generate a timing profile that displays only the time
    spent in these two subroutines, in a flattened structure so that they
    are easy to compare. To do this, we

    #. Ignore subroutines of ``"c"`` and ``"f"`` that are unnecessary for    this comparison

    #. Flatten the hierarchical timing information

    #. Eliminate all the information we don't care about

    >>> import time
    >>> from pyomo.common.timing import HierarchicalTimer
    >>> timer = HierarchicalTimer()
    >>> timer.start("root")
    >>> timer.start("a")
    >>> time.sleep(0.01)
    >>> timer.start("b")
    >>> timer.start("c")
    >>> time.sleep(0.1)
    >>> timer.stop("c")
    >>> timer.stop("b")
    >>> timer.stop("a")
    >>> timer.start("d")
    >>> timer.start("e")
    >>> time.sleep(0.01)
    >>> timer.start("f")
    >>> time.sleep(0.05)
    >>> timer.stop("f")
    >>> timer.start("c")
    >>> timer.start("g")
    >>> timer.start("h")
    >>> time.sleep(0.1)
    >>> timer.stop("h")
    >>> timer.stop("g")
    >>> timer.stop("c")
    >>> timer.stop("e")
    >>> timer.stop("d")
    >>> timer.stop("root")
    >>> print(timer) # doctest: +SKIP
    Identifier                       ncalls   cumtime   percall      %
    ------------------------------------------------------------------
    root                                  1     0.290     0.290  100.0
         -------------------------------------------------------------
         a                                1     0.118     0.118   40.5
              --------------------------------------------------------
              b                           1     0.105     0.105   89.4
                   ---------------------------------------------------
                   c                      1     0.105     0.105  100.0
                   other                n/a     0.000       n/a    0.0
                   ===================================================
              other                     n/a     0.013       n/a   10.6
              ========================================================
         d                                1     0.173     0.173   59.5
              --------------------------------------------------------
              e                           1     0.173     0.173  100.0
                   ---------------------------------------------------
                   c                      1     0.105     0.105   60.9
                        ----------------------------------------------
                        g                 1     0.105     0.105  100.0
                             -----------------------------------------
                             h            1     0.105     0.105  100.0
                             other      n/a     0.000       n/a    0.0
                             =========================================
                        other           n/a     0.000       n/a    0.0
                        ==============================================
                   f                      1     0.055     0.055   31.9
                   other                n/a     0.013       n/a    7.3
                   ===================================================
              other                     n/a     0.000       n/a    0.0
              ========================================================
         other                          n/a     0.000       n/a    0.0
         =============================================================
    ==================================================================
    >>> # Clear subroutines under "c" that we don't care about
    >>> timer.timers["root"].timers["d"].timers["e"].timers["c"].timers.clear()
    >>> # Flatten hierarchy
    >>> timer.timers["root"].flatten()
    >>> # Clear except for the subroutines we care about
    >>> timer.timers["root"].clear_except("c", "f")
    >>> print(timer) # doctest: +SKIP
    Identifier   ncalls   cumtime   percall      %
    ----------------------------------------------
    root              1     0.290     0.290  100.0
         -----------------------------------------
         c            2     0.210     0.105   72.4
         f            1     0.055     0.055   19.0
         other      n/a     0.025       n/a    8.7
         =========================================
    ==============================================

    Notes
    -----

    The :py:class:`HierarchicalTimer` uses a stack to track which timers
    are active at any point in time. Additionally, each timer has a
    dictionary of timers for its children timers. Consider

    >>> timer = HierarchicalTimer()
    >>> timer.start('all')
    >>> timer.start('a')
    >>> timer.start('aa')

    After the above code is run, ``timer.stack`` will be
    ``['all', 'a', 'aa']`` and ``timer.timers`` will have one key,
    ``'all'`` and one value which will be a
    :py:class:`_HierarchicalHelper`. The :py:class:`_HierarchicalHelper`
    has its own timers dictionary:

        ``{'a': _HierarchicalHelper}``

    and so on. This way, we can easily access any timer with something
    that looks like the stack. The logic is recursive (although the
    code is not).

    """

    def __init__(self):
        self.stack = list()
        self.timers = dict()

    def _get_timer(self, identifier, should_exist=False):
        """
        This method gets the timer associated with the current state
        of self.stack and the specified identifier.

        Parameters
        ----------
        identifier: str
            The name of the timer
        should_exist: bool
            The should_exist is True, and the timer does not already
            exist, an error will be raised. If should_exist is False, and
            the timer does not already exist, a new timer will be made.

        Returns
        -------
        timer: _HierarchicalHelper

        """
        parent = self._get_timer_from_stack(self.stack)
        if identifier in parent.timers:
            return parent.timers[identifier]
        else:
            if should_exist:
                raise RuntimeError('Could not find timer {0}'.format('.'.join(self.stack + [identifier])))
            parent.timers[identifier] = _HierarchicalHelper()
            return parent.timers[identifier]

    def start(self, identifier):
        """Start incrementing the timer identified with identifier

        Parameters
        ----------
        identifier: str
            The name of the timer
        """
        timer = self._get_timer(identifier)
        timer.start()
        self.stack.append(identifier)

    def stop(self, identifier):
        """Stop incrementing the timer identified with identifier

        Parameters
        ----------
        identifier: str
            The name of the timer
        """
        if identifier != self.stack[-1]:
            raise ValueError(str(identifier) + ' is not the currently active timer.  The only timer that can currently be stopped is ' + '.'.join(self.stack))
        self.stack.pop()
        timer = self._get_timer(identifier, should_exist=True)
        timer.stop()

    def _get_identifier_len(self):
        stage_timers = list(self.timers.items())
        stage_lengths = list()
        while len(stage_timers) > 0:
            new_stage_timers = list()
            max_len = 0
            for identifier, timer in stage_timers:
                new_stage_timers.extend(timer.timers.items())
                if len(identifier) > max_len:
                    max_len = len(identifier)
            stage_lengths.append(max(max_len, len('other')))
            stage_timers = new_stage_timers
        return stage_lengths

    def __str__(self):
        const_indent = 4
        max_name_length = 200 - 36
        stage_identifier_lengths = self._get_identifier_len()
        name_field_width = sum(stage_identifier_lengths)
        if name_field_width > max_name_length:
            name_field_width = max((const_indent * i + l for i, l in enumerate(stage_identifier_lengths)))
            for i in range(len(stage_identifier_lengths) - 1):
                stage_identifier_lengths[i] = const_indent
            stage_identifier_lengths[-1] = name_field_width - const_indent * (len(stage_identifier_lengths) - 1)
        name_formatter = '{name:<' + str(name_field_width) + '}'
        s = (name_formatter + '{ncalls:>9} {cumtime:>9} {percall:>9} {percent:>6}\n').format(name='Identifier', ncalls='ncalls', cumtime='cumtime', percall='percall', percent='%')
        underline = '-' * (name_field_width + 36) + '\n'
        s += underline
        sub_stage_identifier_lengths = stage_identifier_lengths[1:]
        for name, timer in sorted(self.timers.items()):
            s += (name_formatter + '{ncalls:>9d} {cumtime:>9.3f} {percall:>9.3f} {percent:>6.1f}\n').format(name=name, ncalls=timer.n_calls, cumtime=timer.total_time, percall=timer.total_time / timer.n_calls, percent=self.get_total_percent_time(name))
            s += timer.to_str(indent=' ' * stage_identifier_lengths[0], stage_identifier_lengths=sub_stage_identifier_lengths)
        s += underline.replace('-', '=')
        return s

    def reset(self):
        """
        Completely reset the timer.
        """
        self.stack = list()
        self.timers = dict()

    def _get_timer_from_stack(self, stack):
        """
        This method gets the timer associated with stack.

        Parameters
        ----------
        stack: list of str
            A list of identifiers.

        Returns
        -------
        timer: _HierarchicalHelper
        """
        tmp = self
        for i in stack:
            tmp = tmp.timers[i]
        return tmp

    def get_total_time(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        total_time: float
            The total time spent with the specified timer active.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        return timer.total_time

    def get_num_calls(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        n_calls: int
            The number of times start was called for the specified timer.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        return timer.n_calls

    def get_relative_percent_time(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        percent_time: float
            The percent of time spent in the specified timer
            relative to the timer's immediate parent.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        parent = self._get_timer_from_stack(stack[:-1])
        if parent is self:
            return self.get_total_percent_time(identifier)
        elif parent.total_time > 0:
            return timer.total_time / parent.total_time * 100
        else:
            return float('nan')

    def get_total_percent_time(self, identifier):
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        percent_time: float
            The percent of time spent in the specified timer
            relative to the total time in all timers.
        """
        stack = identifier.split('.')
        timer = self._get_timer_from_stack(stack)
        total_time = 0
        for _timer in self.timers.values():
            total_time += _timer.total_time
        if total_time > 0:
            return timer.total_time / total_time * 100
        else:
            return float('nan')

    def get_timers(self):
        """
        Returns
        -------
        identifiers: list of str
            Returns a list of all timer identifiers
        """
        res = list()
        for name, timer in self.timers.items():
            res.append(name)
            timer.get_timers(res, name)
        return res

    def flatten(self):
        """Flatten the HierarchicalTimer in-place, moving all the timing
        categories into a single level

        If any timers moved into the same level have the same identifier,
        the ``total_time`` and ``n_calls`` fields are added together.
        The ``total_time`` of a "child timer" that is "moved upwards" is
        subtracted from the ``total_time`` of that timer's original
        parent.

        """
        if self.stack:
            raise RuntimeError('Cannot flatten a HierarchicalTimer while any timers are active. Current active timer is %s. flatten should only be called as a post-processing step.' % self.stack[-1])
        items = list(self.timers.items())
        for key, timer in items:
            timer.flatten()
            _move_grandchildren_to_root(self, timer)

    def clear_except(self, *args):
        """Prune all "sub-timers" except those specified

        Parameters
        ----------
        args: str
            Keys that will be retained

        """
        if self.stack:
            raise RuntimeError('Cannot clear a HierarchicalTimer while any timers are active. Current active timer is %s. clear_except should only be called as a post-processing step.' % self.stack[-1])
        to_retain = set(args)
        _clear_timers_except(self, to_retain)