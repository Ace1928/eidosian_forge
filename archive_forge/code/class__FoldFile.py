from __future__ import print_function
import os
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from .utils import make_dirs_if_not_exists
class _FoldFile(FoldStorage):
    """
    FoldFile is the realisation of the interface of FoldStorage. It always saves data to file before reset them.
    All files place to the special directory 'folds'.
    """

    def __init__(self, fold, storage_name, sep, column_description):
        super(_FoldFile, self).__init__(fold, storage_name, sep=sep, column_description=column_description)
        self._file_path = os.path.join(self.default_dir, storage_name)
        self._prepare_path()
        self._lines = []
        self._file = None

    def _prepare_path(self):
        make_dirs_if_not_exists(self.default_dir)
        open(self._file_path, 'w').close()

    def path(self):
        return self._file_path

    def add(self, line):
        self._size += 1
        print(line, file=self._file, end='')

    def add_all(self, lines):
        [self.add(line) for line in lines]

    def open(self):
        if self._file is None:
            self._file = open(self._file_path, mode='a')
        else:
            raise CatBoostError('File already opened {}'.format(self._file_path))

    def is_opened(self):
        return self._file is not None

    def close(self):
        if self._file is None:
            raise CatBoostError('Trying to close None {}'.format(self._file_path))
        self._file.close()
        self._file = None

    def delete(self):
        if self._file is not None:
            raise CatBoostError('Close file before delete')
        if os.path.exists(self._file_path):
            os.remove(self._file_path)