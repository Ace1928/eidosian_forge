import os
import sys
from troveclient.compat import common
class DatastoreConfigurationParameters(common.AuthedCommandsBase):
    """Command to show configuration parameters for a datastore."""
    params = ['datastore', 'parameter']

    def parameters(self):
        """List parameters that can be set."""
        self._require('datastore')
        self._pretty_print(self.dbaas.configuration_parameters.parameters, self.datastore)

    def get_parameter(self):
        """List parameters that can be set."""
        self._require('datastore', 'parameter')
        self._pretty_print(self.dbaas.configuration_parameters.get_parameter, self.datastore, self.parameter)