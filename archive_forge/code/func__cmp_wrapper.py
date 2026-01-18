from typing import Any, Iterable
from .version import __version__ as internal_version
from ._vendor.packaging.version import Version, InvalidVersion
def _cmp_wrapper(self, cmp: Any, method: str) -> bool:
    try:
        return getattr(Version(self), method)(self._convert_to_version(cmp))
    except BaseException as e:
        if not isinstance(e, InvalidVersion):
            raise
        return getattr(super(), method)(cmp)