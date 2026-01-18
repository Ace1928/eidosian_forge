import datetime
import math
import typing as t
from wandb.util import (
def _json_obj_to_params_obj(json_obj: t.Any, artifact: t.Optional['Artifact']=None) -> t.Any:
    """Helper method."""
    if json_obj.__class__ == dict:
        if 'wb_type' in json_obj:
            return TypeRegistry.type_from_dict(json_obj, artifact)
        else:
            return {key: _json_obj_to_params_obj(json_obj[key], artifact) for key in json_obj}
    elif json_obj.__class__ == list:
        return [_json_obj_to_params_obj(item, artifact) for item in json_obj]
    else:
        return json_obj