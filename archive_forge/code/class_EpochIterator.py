import warnings
from keras.src.trainers import data_adapters
class EpochIterator:

    def __init__(self, x, y=None, sample_weight=None, batch_size=None, steps_per_epoch=None, shuffle=False, class_weight=None, steps_per_execution=1):
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_execution = steps_per_execution
        if steps_per_epoch:
            self._current_iterator = None
            self._insufficient_data = False
        self.data_adapter = data_adapters.get_data_adapter(x=x, y=y, sample_weight=sample_weight, batch_size=batch_size, steps_per_epoch=steps_per_epoch, shuffle=shuffle, class_weight=class_weight)
        self._num_batches = self.data_adapter.num_batches

    def _get_iterator(self):
        return self.data_adapter.get_numpy_iterator()

    def enumerate_epoch(self):
        buffer = []
        if self.steps_per_epoch:
            if self._current_iterator is None:
                self._current_iterator = iter(self._get_iterator())
                self._insufficient_data = False
            for step in range(self.steps_per_epoch):
                if self._insufficient_data:
                    break
                try:
                    data = next(self._current_iterator)
                    buffer.append(data)
                    if len(buffer) == self.steps_per_execution:
                        yield (step - len(buffer) + 1, buffer)
                        buffer = []
                except (StopIteration,):
                    warnings.warn('Your input ran out of data; interrupting epoch. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.', stacklevel=2)
                    self._current_iterator = None
                    self._insufficient_data = True
            if buffer:
                yield (step - len(buffer) + 1, buffer)
        else:
            for step, data in enumerate(self._get_iterator()):
                buffer.append(data)
                if len(buffer) == self.steps_per_execution:
                    yield (step - len(buffer) + 1, buffer)
                    buffer = []
            if buffer:
                yield (step - len(buffer) + 1, buffer)
            if not self._num_batches:
                self._num_batches = step + 1
        self.data_adapter.on_epoch_end()

    @property
    def num_batches(self):
        if self.steps_per_epoch:
            return self.steps_per_epoch
        return self._num_batches