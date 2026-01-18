import abc
class HealthcheckBaseExtension(metaclass=abc.ABCMeta):

    def __init__(self, oslo_conf, conf):
        self.oslo_conf = oslo_conf
        self.conf = conf

    @abc.abstractmethod
    def healthcheck(self, server_port):
        """method called by the healthcheck middleware

        return: HealthcheckResult object
        """

    def _conf_get(self, key, group='healthcheck'):
        if key in self.conf:
            self.oslo_conf.set_override(key, self.conf[key], group=group)
        return getattr(getattr(self.oslo_conf, group), key)