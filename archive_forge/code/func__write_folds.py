import random
from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile
def _write_folds(self, fold_storages, num, offset):
    """Learn_set contains numbers of lines. The method itself store relevant lines from dataset to fold storage."""
    generator = self._line_reader.lines_generator()
    for fold_storage in fold_storages:
        fold_storage.open()
    try:
        rest_folds = []
        rest_fold_file = self.create_fold(None, 'offset{}_rest'.format(offset), num)
        rest_fold_file.open()
        num += 1
        rest_size = 0
        for num_line, (_, line) in enumerate(generator):
            group_id = self._line_groups_ids[num_line]
            is_written = False
            for fold_storage in fold_storages:
                if fold_storage.contains_group_id(group_id):
                    fold_storage.add(line)
                    is_written = True
            if not is_written:
                rest_fold_file.add(line)
                rest_size += 1
                if rest_size >= self._REST_SIZE:
                    rest_folds.append(rest_fold_file)
                    rest_fold_file.close()
                    rest_fold_file = self.create_fold(None, 'offset{}_rest'.format(offset), num)
                    rest_fold_file.open()
                    rest_size = 0
                    num += 1
        if rest_size > 0:
            rest_fold_file.close()
            rest_folds.append(rest_fold_file)
        elif rest_fold_file.is_opened():
            rest_fold_file.close()
    finally:
        for fold_storage in fold_storages:
            fold_storage.close()
    return rest_folds