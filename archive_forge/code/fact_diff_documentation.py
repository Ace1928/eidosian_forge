from __future__ import absolute_import, division, print_function
import re
from ansible.plugins.callback import CallbackBase
from ansible.errors import AnsibleFilterError
Compare two facts or variables and get a diff.
    :param before: The first fact to be used in the comparison.
    :type before: raw
    :param after: The second fact to be used in the comparison.
    :type after: raw
    :param plugin: The name of the plugin in collection format
    :type plugin: string
    