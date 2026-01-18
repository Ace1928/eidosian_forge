import os
import shutil
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.hashutil import md5_file_hex
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.wb_value import WBValue
class _PicklingSavedModel(_SavedModel[SavedModelObjType]):
    _dep_py_files: Optional[List[str]] = None
    _dep_py_files_path: Optional[str] = None

    def __init__(self, obj_or_path: Union[SavedModelObjType, str], dep_py_files: Optional[List[str]]=None):
        super().__init__(obj_or_path)
        if self.__class__ == _PicklingSavedModel:
            raise TypeError('Cannot instantiate abstract _PicklingSavedModel class - please use SavedModel.init(...) instead.')
        if dep_py_files is not None and len(dep_py_files) > 0:
            self._dep_py_files = dep_py_files
            self._dep_py_files_path = os.path.abspath(os.path.join(MEDIA_TMP.name, runid.generate_id()))
            os.makedirs(self._dep_py_files_path, exist_ok=True)
            for extra_file in self._dep_py_files:
                if os.path.isfile(extra_file):
                    shutil.copy(extra_file, self._dep_py_files_path)
                elif os.path.isdir(extra_file):
                    shutil.copytree(extra_file, os.path.join(self._dep_py_files_path, os.path.basename(extra_file)))
                else:
                    raise ValueError(f'Invalid dependency file: {extra_file}')

    @classmethod
    def from_json(cls: Type['_SavedModel'], json_obj: dict, source_artifact: 'Artifact') -> '_PicklingSavedModel':
        backup_path = [p for p in sys.path]
        if 'dep_py_files_path' in json_obj and json_obj['dep_py_files_path'] is not None:
            dl_path = _load_dir_from_artifact(source_artifact, json_obj['dep_py_files_path'])
            assert dl_path is not None
            sys.path.append(dl_path)
        inst = super().from_json(json_obj, source_artifact)
        sys.path = backup_path
        return inst

    def to_json(self, run_or_artifact: Union['LocalRun', 'Artifact']) -> dict:
        json_obj = super().to_json(run_or_artifact)
        assert isinstance(run_or_artifact, wandb.Artifact)
        if self._dep_py_files_path is not None:
            json_obj['dep_py_files_path'] = _add_deterministic_dir_to_artifact(run_or_artifact, self._dep_py_files_path, os.path.join('.wb_data', 'extra_files'))
        return json_obj