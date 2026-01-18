from dataclasses import dataclass
from typing import Dict, Optional, Tuple
@dataclass(frozen=True)
class SelectiveBuildOperator:
    name: str
    is_root_operator: bool
    is_used_for_training: bool
    include_all_overloads: bool
    _debug_info: Optional[Tuple[str, ...]]

    @staticmethod
    def from_yaml_dict(op_name: str, op_info: Dict[str, object]) -> 'SelectiveBuildOperator':
        allowed_keys = {'name', 'is_root_operator', 'is_used_for_training', 'include_all_overloads', 'debug_info'}
        if len(set(op_info.keys()) - allowed_keys) > 0:
            raise Exception('Got unexpected top level keys: {}'.format(','.join(set(op_info.keys()) - allowed_keys)))
        if 'name' in op_info:
            assert op_name == op_info['name']
        is_root_operator = op_info.get('is_root_operator', True)
        assert isinstance(is_root_operator, bool)
        is_used_for_training = op_info.get('is_used_for_training', True)
        assert isinstance(is_used_for_training, bool)
        include_all_overloads = op_info.get('include_all_overloads', True)
        assert isinstance(include_all_overloads, bool)
        debug_info: Optional[Tuple[str, ...]] = None
        if 'debug_info' in op_info:
            di_list = op_info['debug_info']
            assert isinstance(di_list, list)
            debug_info = tuple((str(x) for x in di_list))
        return SelectiveBuildOperator(name=op_name, is_root_operator=is_root_operator, is_used_for_training=is_used_for_training, include_all_overloads=include_all_overloads, _debug_info=debug_info)

    @staticmethod
    def from_legacy_operator_name_without_overload(name: str) -> 'SelectiveBuildOperator':
        return SelectiveBuildOperator(name=name, is_root_operator=True, is_used_for_training=True, include_all_overloads=True, _debug_info=None)

    def to_dict(self) -> Dict[str, object]:
        ret: Dict[str, object] = {'is_root_operator': self.is_root_operator, 'is_used_for_training': self.is_used_for_training, 'include_all_overloads': self.include_all_overloads}
        if self._debug_info is not None:
            ret['debug_info'] = self._debug_info
        return ret