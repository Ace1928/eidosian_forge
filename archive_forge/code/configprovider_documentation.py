import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
Lookup the configured endpoint URL.

        The order is:

        1. The value provided by a service-specific environment variable.
        2. The value provided by the global endpoint environment variable
           (AWS_ENDPOINT_URL).
        3. The value provided by a service-specific parameter from a services
           definition section in the shared configuration file.
        4. The value provided by the global parameter from a services
           definition section in the shared configuration file.
        