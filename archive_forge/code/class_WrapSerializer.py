from __future__ import annotations
import dataclasses
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class WrapSerializer:
    """Wrap serializers receive the raw inputs along with a handler function that applies the standard serialization
    logic, and can modify the resulting value before returning it as the final output of serialization.

    For example, here's a scenario in which a wrap serializer transforms timezones to UTC **and** utilizes the existing `datetime` serialization logic.

    ```python
    from datetime import datetime, timezone
    from typing import Any, Dict

    from typing_extensions import Annotated

    from pydantic import BaseModel, WrapSerializer

    class EventDatetime(BaseModel):
        start: datetime
        end: datetime

    def convert_to_utc(value: Any, handler, info) -> Dict[str, datetime]:
        # Note that `helper` can actually help serialize the `value` for further custom serialization in case it's a subclass.
        partial_result = handler(value, info)
        if info.mode == 'json':
            return {
                k: datetime.fromisoformat(v).astimezone(timezone.utc)
                for k, v in partial_result.items()
            }
        return {k: v.astimezone(timezone.utc) for k, v in partial_result.items()}

    UTCEventDatetime = Annotated[EventDatetime, WrapSerializer(convert_to_utc)]

    class EventModel(BaseModel):
        event_datetime: UTCEventDatetime

    dt = EventDatetime(
        start='2024-01-01T07:00:00-08:00', end='2024-01-03T20:00:00+06:00'
    )
    event = EventModel(event_datetime=dt)
    print(event.model_dump())
    '''
    {
        'event_datetime': {
            'start': datetime.datetime(
                2024, 1, 1, 15, 0, tzinfo=datetime.timezone.utc
            ),
            'end': datetime.datetime(
                2024, 1, 3, 14, 0, tzinfo=datetime.timezone.utc
            ),
        }
    }
    '''

    print(event.model_dump_json())
    '''
    {"event_datetime":{"start":"2024-01-01T15:00:00Z","end":"2024-01-03T14:00:00Z"}}
    '''
    ```

    Attributes:
        func: The serializer function to be wrapped.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
        when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
            `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.
    """
    func: core_schema.WrapSerializerFunction
    return_type: Any = PydanticUndefined
    when_used: Literal['always', 'unless-none', 'json', 'json-unless-none'] = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """This method is used to get the Pydantic core schema of the class.

        Args:
            source_type: Source type.
            handler: Core schema handler.

        Returns:
            The generated core schema of the class.
        """
        schema = handler(source_type)
        try:
            return_type = _decorators.get_function_return_type(self.func, self.return_type, handler._get_types_namespace())
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(function=self.func, info_arg=_decorators.inspect_annotated_serializer(self.func, 'wrap'), return_schema=return_schema, when_used=self.when_used)
        return schema