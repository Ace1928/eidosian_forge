import abc
import dataclasses
import functools
import inspect
import sys
from dataclasses import Field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union, Type, get_type_hints
from enum import Enum
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.utils import CatchAllVar
class _CatchAllUndefinedParameters(_UndefinedParameterAction):
    """
    This class allows to add a field of type utils.CatchAll which acts as a
    dictionary into which all
    undefined parameters will be written.
    These parameters are not affected by LetterCase.
    If no undefined parameters are given, this dictionary will be empty.
    """

    class _SentinelNoDefault:
        pass

    @staticmethod
    def handle_from_dict(cls, kvs: Dict) -> Dict[str, Any]:
        known, unknown = _UndefinedParameterAction._separate_defined_undefined_kvs(cls=cls, kvs=kvs)
        catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(cls=cls)
        if catch_all_field.name in known:
            already_parsed = isinstance(known[catch_all_field.name], dict)
            default_value = _CatchAllUndefinedParameters._get_default(catch_all_field=catch_all_field)
            received_default = default_value == known[catch_all_field.name]
            value_to_write: Any
            if received_default and len(unknown) == 0:
                value_to_write = default_value
            elif received_default and len(unknown) > 0:
                value_to_write = unknown
            elif already_parsed:
                value_to_write = known[catch_all_field.name]
                if len(unknown) > 0:
                    value_to_write.update(unknown)
            else:
                error_message = f"Received input field with same name as catch-all field: '{catch_all_field.name}': '{known[catch_all_field.name]}'"
                raise UndefinedParameterError(error_message)
        else:
            value_to_write = unknown
        known[catch_all_field.name] = value_to_write
        return known

    @staticmethod
    def _get_default(catch_all_field: Field) -> Any:
        has_default = not isinstance(catch_all_field.default, dataclasses._MISSING_TYPE)
        has_default_factory = not isinstance(catch_all_field.default_factory, dataclasses._MISSING_TYPE)
        default_value: Union[Type[_CatchAllUndefinedParameters._SentinelNoDefault], Any] = _CatchAllUndefinedParameters._SentinelNoDefault
        if has_default:
            default_value = catch_all_field.default
        elif has_default_factory:
            default_value = catch_all_field.default_factory()
        return default_value

    @staticmethod
    def handle_to_dict(obj, kvs: Dict[Any, Any]) -> Dict[Any, Any]:
        catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(obj.__class__)
        undefined_parameters = kvs.pop(catch_all_field.name)
        if isinstance(undefined_parameters, dict):
            kvs.update(undefined_parameters)
        return kvs

    @staticmethod
    def handle_dump(obj) -> Dict[Any, Any]:
        catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(cls=obj)
        return getattr(obj, catch_all_field.name)

    @staticmethod
    def create_init(obj) -> Callable:
        original_init = obj.__init__
        init_signature = inspect.signature(original_init)

        @functools.wraps(obj.__init__)
        def _catch_all_init(self, *args, **kwargs):
            known_kwargs, unknown_kwargs = _CatchAllUndefinedParameters._separate_defined_undefined_kvs(obj, kwargs)
            num_params_takeable = len(init_signature.parameters) - 1
            if _CatchAllUndefinedParameters._get_catch_all_field(obj).name not in known_kwargs:
                num_params_takeable -= 1
            num_args_takeable = num_params_takeable - len(known_kwargs)
            args, unknown_args = (args[:num_args_takeable], args[num_args_takeable:])
            bound_parameters = init_signature.bind_partial(self, *args, **known_kwargs)
            unknown_args = {f'_UNKNOWN{i}': v for i, v in enumerate(unknown_args)}
            arguments = bound_parameters.arguments
            arguments.update(unknown_args)
            arguments.update(unknown_kwargs)
            arguments.pop('self', None)
            final_parameters = _CatchAllUndefinedParameters.handle_from_dict(obj, arguments)
            original_init(self, **final_parameters)
        return _catch_all_init

    @staticmethod
    def _get_catch_all_field(cls) -> Field:
        cls_globals = vars(sys.modules[cls.__module__])
        types = get_type_hints(cls, globalns=cls_globals)
        catch_all_fields = list(filter(lambda f: types[f.name] == Optional[CatchAllVar], fields(cls)))
        number_of_catch_all_fields = len(catch_all_fields)
        if number_of_catch_all_fields == 0:
            raise UndefinedParameterError('No field of type dataclasses_json.CatchAll defined')
        elif number_of_catch_all_fields > 1:
            raise UndefinedParameterError(f'Multiple catch-all fields supplied: {number_of_catch_all_fields}.')
        else:
            return catch_all_fields[0]