import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def _id_matches(self, other):
    my_id = self.component_id
    other_id = other.component_id
    self_dict = isinstance(my_id, dict)
    other_dict = isinstance(other_id, dict)
    if self_dict != other_dict:
        return False
    if self_dict:
        if set(my_id.keys()) != set(other_id.keys()):
            return False
        for k, v in my_id.items():
            other_v = other_id[k]
            if v == other_v:
                continue
            v_wild = isinstance(v, _Wildcard)
            other_wild = isinstance(other_v, _Wildcard)
            if v_wild or other_wild:
                if not (v_wild and other_wild):
                    continue
                if v is ALL or other_v is ALL:
                    continue
                if v is MATCH or other_v is MATCH:
                    return False
            else:
                return False
        return True
    return my_id == other_id