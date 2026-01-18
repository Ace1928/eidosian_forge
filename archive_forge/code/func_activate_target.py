from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def activate_target(target_id: TargetID) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Activates (focuses) the target.

    :param target_id:
    """
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.activateTarget', 'params': params}
    json = (yield cmd_dict)