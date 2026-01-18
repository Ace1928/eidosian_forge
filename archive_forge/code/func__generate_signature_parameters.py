from __future__ import annotations
import dataclasses
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import PydanticUndefined
from ._config import ConfigWrapper
from ._utils import is_valid_identifier
def _generate_signature_parameters(init: Callable[..., None], fields: dict[str, FieldInfo], config_wrapper: ConfigWrapper) -> dict[str, Parameter]:
    """Generate a mapping of parameter names to Parameter objects for a pydantic BaseModel or dataclass."""
    from itertools import islice
    present_params = signature(init).parameters.values()
    merged_params: dict[str, Parameter] = {}
    var_kw = None
    use_var_kw = False
    for param in islice(present_params, 1, None):
        if fields.get(param.name):
            if getattr(fields[param.name], 'init', True) is False:
                continue
            param = param.replace(name=_field_name_for_signature(param.name, fields[param.name]))
        if param.annotation == 'Any':
            param = param.replace(annotation=Any)
        if param.kind is param.VAR_KEYWORD:
            var_kw = param
            continue
        merged_params[param.name] = param
    if var_kw:
        allow_names = config_wrapper.populate_by_name
        for field_name, field in fields.items():
            param_name = _field_name_for_signature(field_name, field)
            if field_name in merged_params or param_name in merged_params:
                continue
            if not is_valid_identifier(param_name):
                if allow_names:
                    param_name = field_name
                else:
                    use_var_kw = True
                    continue
            kwargs = {} if field.is_required() else {'default': field.get_default(call_default_factory=False)}
            merged_params[param_name] = Parameter(param_name, Parameter.KEYWORD_ONLY, annotation=field.rebuild_annotation(), **kwargs)
    if config_wrapper.extra == 'allow':
        use_var_kw = True
    if var_kw and use_var_kw:
        default_model_signature = [('self', Parameter.POSITIONAL_ONLY), ('data', Parameter.VAR_KEYWORD)]
        if [(p.name, p.kind) for p in present_params] == default_model_signature:
            var_kw_name = 'extra_data'
        else:
            var_kw_name = var_kw.name
        while var_kw_name in fields:
            var_kw_name += '_'
        merged_params[var_kw_name] = var_kw.replace(name=var_kw_name)
    return merged_params