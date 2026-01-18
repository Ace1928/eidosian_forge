import abc
import typing as t
from .interface.summary_record import SummaryItem, SummaryRecord
class SummaryDict(metaclass=abc.ABCMeta):
    """dict-like wrapper for the nested dictionaries in a SummarySubDict.

    Triggers self._root._callback on property changes.
    """

    @abc.abstractmethod
    def _as_dict(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _update(self, record: SummaryRecord):
        raise NotImplementedError

    def keys(self):
        return [k for k in self._as_dict().keys() if k != '_wandb']

    def get(self, key, default=None):
        return self._as_dict().get(key, default)

    def __getitem__(self, key):
        item = self._as_dict()[key]
        if isinstance(item, dict):
            wrapped_item = SummarySubDict()
            object.__setattr__(wrapped_item, '_items', item)
            object.__setattr__(wrapped_item, '_parent', self)
            object.__setattr__(wrapped_item, '_parent_key', key)
            return wrapped_item
        return item
    __getattr__ = __getitem__

    def __setitem__(self, key, val):
        self.update({key: val})
    __setattr__ = __setitem__

    def __delattr__(self, key):
        record = SummaryRecord()
        item = SummaryItem()
        item.key = (key,)
        record.remove = (item,)
        self._update(record)
    __delitem__ = __delattr__

    def update(self, d: t.Dict):
        record = SummaryRecord()
        for key, value in d.items():
            item = SummaryItem()
            item.key = (key,)
            item.value = value
            record.update.append(item)
        self._update(record)