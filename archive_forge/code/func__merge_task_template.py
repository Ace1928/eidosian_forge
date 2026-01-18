from .. import auth, errors, utils
from ..types import ServiceMode
def _merge_task_template(current, override):
    merged = current.copy()
    if override is not None:
        for ts_key, ts_value in override.items():
            if ts_key == 'ContainerSpec':
                if 'ContainerSpec' not in merged:
                    merged['ContainerSpec'] = {}
                for cs_key, cs_value in override['ContainerSpec'].items():
                    if cs_value is not None:
                        merged['ContainerSpec'][cs_key] = cs_value
            elif ts_value is not None:
                merged[ts_key] = ts_value
    return merged