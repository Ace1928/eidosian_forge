import os
import socket
import time
from oslo_serialization import jsonutils
import tenacity
from octavia_lib.api.drivers import data_models
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
Get a L7 rule object.

        :param l7rule_id: The L7 rule ID to lookup.
        :type l7rule_id: UUID string
        :raises DriverAgentTimeout: The driver agent did not respond
          inside the timeout.
        :raises DriverError: An unexpected error occurred.
        :returns: A L7Rule object or None if not found.
        