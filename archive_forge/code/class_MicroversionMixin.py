import abc
from heat.common import exception
class MicroversionMixin(object, metaclass=abc.ABCMeta):
    """Mixin For microversion support."""

    def client(self, version=None):
        if version is None:
            version = self.get_max_microversion()
        elif not self.is_version_supported(version):
            raise exception.InvalidServiceVersion(version=version, service=self._get_service_name())
        if version in self._client_instances:
            return self._client_instances[version]
        self._client_instances[version] = self._create(version=version)
        return self._client_instances[version]

    @abc.abstractmethod
    def get_max_microversion(self):
        pass

    @abc.abstractmethod
    def is_version_supported(self, version):
        pass