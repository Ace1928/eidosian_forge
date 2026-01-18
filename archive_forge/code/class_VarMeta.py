from __future__ import absolute_import, division, print_function
import copy
class VarMeta(object):
    """
    DEPRECATION WARNING

    This class is deprecated and will be removed in community.general 10.0.0
    Modules should use the VarDict from plugins/module_utils/vardict.py instead.
    """
    NOTHING = object()

    def __init__(self, diff=False, output=True, change=None, fact=False):
        self.init = False
        self.initial_value = None
        self.value = None
        self.diff = diff
        self.change = diff if change is None else change
        self.output = output
        self.fact = fact

    def set(self, diff=None, output=None, change=None, fact=None, initial_value=NOTHING):
        if diff is not None:
            self.diff = diff
        if output is not None:
            self.output = output
        if change is not None:
            self.change = change
        if fact is not None:
            self.fact = fact
        if initial_value is not self.NOTHING:
            self.initial_value = copy.deepcopy(initial_value)

    def set_value(self, value):
        if not self.init:
            self.initial_value = copy.deepcopy(value)
            self.init = True
        self.value = value
        return self

    @property
    def has_changed(self):
        return self.change and self.initial_value != self.value

    @property
    def diff_result(self):
        return None if not (self.diff and self.has_changed) else {'before': self.initial_value, 'after': self.value}

    def __str__(self):
        return '<VarMeta: value={0}, initial={1}, diff={2}, output={3}, change={4}>'.format(self.value, self.initial_value, self.diff, self.output, self.change)