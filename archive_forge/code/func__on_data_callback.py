from kivy.properties import ListProperty, ObservableDict, ObjectProperty
from kivy.event import EventDispatcher
from functools import partial
def _on_data_callback(self, instance, value):
    last_len = self._last_len
    new_len = self._last_len = len(self.data)
    op, val = value.last_op
    if op == '__setitem__':
        val = recondition_slice_assign(val, last_len, new_len)
        if val is not None:
            self.dispatch('on_data_changed', modified=val)
        else:
            self.dispatch('on_data_changed')
    elif op == '__delitem__':
        self.dispatch('on_data_changed', removed=val)
    elif op == '__setslice__':
        val = recondition_slice_assign(slice(*val), last_len, new_len)
        if val is not None:
            self.dispatch('on_data_changed', modified=val)
        else:
            self.dispatch('on_data_changed')
    elif op == '__delslice__':
        self.dispatch('on_data_changed', removed=slice(*val))
    elif op == '__iadd__' or op == '__imul__':
        self.dispatch('on_data_changed', appended=slice(last_len, new_len))
    elif op == 'append':
        self.dispatch('on_data_changed', appended=slice(last_len, new_len))
    elif op == 'insert':
        self.dispatch('on_data_changed', inserted=val)
    elif op == 'pop':
        if val:
            self.dispatch('on_data_changed', removed=val[0])
        else:
            self.dispatch('on_data_changed', removed=last_len - 1)
    elif op == 'extend':
        self.dispatch('on_data_changed', appended=slice(last_len, new_len))
    else:
        self.dispatch('on_data_changed')