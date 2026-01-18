import abc
import re
import typing
class retry_if_exception_message(retry_if_exception):
    """Retries if an exception message equals or matches."""

    def __init__(self, message: typing.Optional[str]=None, match: typing.Optional[str]=None) -> None:
        if message and match:
            raise TypeError(f"{self.__class__.__name__}() takes either 'message' or 'match', not both")
        if message:

            def message_fnc(exception: BaseException) -> bool:
                return message == str(exception)
            predicate = message_fnc
        elif match:
            prog = re.compile(match)

            def match_fnc(exception: BaseException) -> bool:
                return bool(prog.match(str(exception)))
            predicate = match_fnc
        else:
            raise TypeError(f"{self.__class__.__name__}() missing 1 required argument 'message' or 'match'")
        super().__init__(predicate)