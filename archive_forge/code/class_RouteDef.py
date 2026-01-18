import abc
import os  # noqa
from typing import (
import attr
from . import hdrs
from .abc import AbstractView
from .typedefs import Handler, PathLike
@attr.s(auto_attribs=True, frozen=True, repr=False, slots=True)
class RouteDef(AbstractRouteDef):
    method: str
    path: str
    handler: _HandlerType
    kwargs: Dict[str, Any]

    def __repr__(self) -> str:
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f', {name}={value!r}')
        return '<RouteDef {method} {path} -> {handler.__name__!r}{info}>'.format(method=self.method, path=self.path, handler=self.handler, info=''.join(info))

    def register(self, router: UrlDispatcher) -> List[AbstractRoute]:
        if self.method in hdrs.METH_ALL:
            reg = getattr(router, 'add_' + self.method.lower())
            return [reg(self.path, self.handler, **self.kwargs)]
        else:
            return [router.add_route(self.method, self.path, self.handler, **self.kwargs)]