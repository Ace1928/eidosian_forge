from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
class LazyJRPC(FastAPI):

    def openapi(self):
        result = super().openapi()
        for route in self.routes:
            if isinstance(route, (EntrypointRoute, MethodRoute)):
                route: Union[EntrypointRoute, MethodRoute]
                for media_type in result['paths'][route.path]:
                    result['paths'][route.path][media_type]['responses'].pop('default', None)
        return result

    def bind_entrypoint(self, ep: Entrypoint):
        ep.bind_dependency_overrides_provider(self)
        self.routes.extend(ep.routes)
        self.on_event('shutdown')(ep.shutdown)