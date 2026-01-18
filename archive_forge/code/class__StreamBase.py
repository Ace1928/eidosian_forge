from abc import ABC, abstractmethod
class _StreamBase(ABC):
    """Base stream class abstraction for multi backends Stream to herit from"""

    @abstractmethod
    def wait_event(self, event):
        raise NotImplementedError()

    @abstractmethod
    def wait_stream(self, stream):
        raise NotImplementedError()

    @abstractmethod
    def record_event(self, event=None):
        raise NotImplementedError()

    @abstractmethod
    def query(self):
        raise NotImplementedError()

    @abstractmethod
    def synchronize(self):
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, stream):
        raise NotImplementedError()