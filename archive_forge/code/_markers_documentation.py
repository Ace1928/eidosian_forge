from typing import TYPE_CHECKING, Any, Callable, TypeVar
from typing_extensions import Annotated
from .. import _singleton
Decorator for applying configuration options.

    Configuration markers are implemented via `typing.Annotated` and straightforward to
    apply to types, for example:

    ```python
    field: tyro.conf.FlagConversionOff[bool]
    ```

    This decorator makes markers applicable to general functions as well:

    ```python
    # Recursively apply FlagConversionOff to all fields in `main()`.
    @tyro.conf.configure(tyro.conf.FlagConversionOff)
    def main(field: bool) -> None:
        ...
    ```

    Args:
        markers: Options to apply.
    