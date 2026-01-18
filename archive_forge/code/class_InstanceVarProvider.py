import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class InstanceVarProvider(BaseProvider):
    """This class loads config values from the session instance vars."""

    def __init__(self, instance_var, session):
        """Initialize InstanceVarProvider.

        :type instance_var: str
        :param instance_var: The instance variable to load from the session.

        :type session: :class:`botocore.session.Session`
        :param session: The botocore session to get the loaded configuration
            file variables from.
        """
        self._instance_var = instance_var
        self._session = session

    def __deepcopy__(self, memo):
        return InstanceVarProvider(copy.deepcopy(self._instance_var, memo), self._session)

    def provide(self):
        """Provide a config value from the session instance vars."""
        instance_vars = self._session.instance_variables()
        value = instance_vars.get(self._instance_var)
        return value

    def __repr__(self):
        return 'InstanceVarProvider(instance_var={}, session={})'.format(self._instance_var, self._session)