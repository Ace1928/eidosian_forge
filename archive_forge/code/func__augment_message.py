import re
from lightning_fabric.utilities.exceptions import MisconfigurationException  # noqa: F401
def _augment_message(exception: BaseException, pattern: str, new_message: str) -> None:
    exception.args = tuple((new_message if isinstance(arg, str) and re.match(pattern, arg, re.DOTALL) else arg for arg in exception.args))