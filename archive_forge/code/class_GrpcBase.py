from __future__ import absolute_import, division, print_function
from functools import wraps
from ansible.plugins import AnsiblePlugin
class GrpcBase(AnsiblePlugin):
    """
    A base class for implementing gRPC abstraction layer
    """
    __rpc__ = ['channel', 'get_config', 'edit_config', 'get']

    def __init__(self, connection):
        super(GrpcBase, self).__init__()
        self._connection = connection

    @property
    @ensure_connect
    def channel(self):
        return self._connection._channel

    def get_config(self, section=None):
        """
        Retrieve all or part of a specified configuration
        (by default entire configuration is retrieved).
        :param section: This argument specifies the portion of the configuration data to retrieve
        :return: Returns the response received from gRPC server from target host in string format
        """
        pass

    def get(self, section=None):
        """
        Retrieve device state information.
        :param section: This argument specifies the portion of the state data to retrieve
                       (by default entire state data is retrieved)
        :return: Returns the json string as a response
        """
        pass

    def edit_config(self, config=None, action=None):
        """
        Loads all or part of the specified *config* to the configuration datastore.
        :param config: The configuration that needs to be push on target host
        :param action: The action to be performed on the configuration datastore for example: 'merge',
                       'replace', 'delete' etc.
        :return: Returns the response received from gRPC server from target host in string format
        """
        pass