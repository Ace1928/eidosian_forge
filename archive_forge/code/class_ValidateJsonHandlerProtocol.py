from __future__ import annotations
from typing import Any, Callable, NamedTuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Literal, Protocol, TypeAlias
class ValidateJsonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    """Event handler for `SchemaValidator.validate_json`."""

    def on_enter(self, input: str | bytes | bytearray, *, strict: bool | None=None, context: dict[str, Any] | None=None, self_instance: Any | None=None) -> None:
        """Callback to be notified of validation start, and create an instance of the event handler.

        Args:
            input: The JSON data to be validated.
            strict: Whether to validate the object in strict mode.
            context: The context to use for validation, this is passed to functional validators.
            self_instance: An instance of a model to set attributes on from validation, this is used when running
                validation from the `__init__` method of a model.
        """
        pass