import inspect
from functools import partial
from itertools import chain
from wandb_promise import Promise
class MiddlewareManager(object):

    def __init__(self, *middlewares, **kwargs):
        self.middlewares = middlewares
        self.wrap_in_promise = kwargs.get('wrap_in_promise', True)
        self._middleware_resolvers = list(get_middleware_resolvers(middlewares))
        self._cached_resolvers = {}

    def get_field_resolver(self, field_resolver):
        if field_resolver not in self._cached_resolvers:
            self._cached_resolvers[field_resolver] = middleware_chain(field_resolver, self._middleware_resolvers, wrap_in_promise=self.wrap_in_promise)
        return self._cached_resolvers[field_resolver]