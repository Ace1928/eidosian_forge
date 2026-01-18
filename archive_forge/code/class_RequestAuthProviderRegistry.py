import warnings
import entrypoints
class RequestAuthProviderRegistry:

    def __init__(self):
        self._registry = []

    def register(self, request_auth_provider):
        self._registry.append(request_auth_provider())

    def register_entrypoints(self):
        for entrypoint in entrypoints.get_group_all(REQUEST_AUTH_PROVIDER_ENTRYPOINT):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn('Failure attempting to register request auth provider "{}": {}'.format(entrypoint.name, str(exc)), stacklevel=2)

    def __iter__(self):
        return iter(self._registry)