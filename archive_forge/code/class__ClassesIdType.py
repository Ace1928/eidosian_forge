import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union
from .. import _dtypes
from ..base_types.media import Media
class _ClassesIdType(_dtypes.Type):
    name = 'classesId'
    legacy_names = ['wandb.Classes_id']
    types = [Classes]

    def __init__(self, classes_obj: Optional[Classes]=None, valid_ids: Optional['_dtypes.UnionType']=None):
        if valid_ids is None:
            valid_ids = _dtypes.UnionType()
        elif isinstance(valid_ids, list):
            valid_ids = _dtypes.UnionType([_dtypes.ConstType(item) for item in valid_ids])
        elif isinstance(valid_ids, _dtypes.UnionType):
            valid_ids = valid_ids
        else:
            raise TypeError('valid_ids must be None, list, or UnionType')
        if classes_obj is None:
            classes_obj = Classes([{'id': _id.params['val'], 'name': str(_id.params['val'])} for _id in valid_ids.params['allowed_types']])
        elif not isinstance(classes_obj, Classes):
            raise TypeError('valid_ids must be None, or instance of Classes')
        else:
            valid_ids = _dtypes.UnionType([_dtypes.ConstType(class_obj['id']) for class_obj in classes_obj._class_set])
        self.wb_classes_obj_ref = classes_obj
        self.params.update({'valid_ids': valid_ids})

    def assign(self, py_obj: Optional[Any]=None) -> '_dtypes.Type':
        return self.assign_type(_dtypes.ConstType(py_obj))

    def assign_type(self, wb_type: '_dtypes.Type') -> '_dtypes.Type':
        valid_ids = self.params['valid_ids'].assign_type(wb_type)
        if not isinstance(valid_ids, _dtypes.InvalidType):
            return self
        return _dtypes.InvalidType()

    @classmethod
    def from_obj(cls, py_obj: Optional[Any]=None) -> '_dtypes.Type':
        return cls(py_obj)

    def to_json(self, artifact: Optional['Artifact']=None) -> Dict[str, Any]:
        cl_dict = super().to_json(artifact)
        if artifact is not None:
            class_name = os.path.join('media', 'cls')
            classes_entry = artifact.add(self.wb_classes_obj_ref, class_name)
            cl_dict['params']['classes_obj'] = {'type': 'classes-file', 'path': classes_entry.path, 'digest': classes_entry.digest}
        else:
            cl_dict['params']['classes_obj'] = self.wb_classes_obj_ref.to_json(artifact)
        return cl_dict

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any], artifact: Optional['Artifact']=None) -> '_dtypes.Type':
        classes_obj = None
        if json_dict.get('params', {}).get('classes_obj', {}).get('type') == 'classes-file':
            if artifact is not None:
                classes_obj = artifact.get(json_dict.get('params', {}).get('classes_obj', {}).get('path'))
                assert classes_obj is None or isinstance(classes_obj, Classes)
            else:
                raise RuntimeError('Expected artifact to be non-null.')
        else:
            classes_obj = Classes.from_json(json_dict['params']['classes_obj'], artifact)
        return cls(classes_obj)