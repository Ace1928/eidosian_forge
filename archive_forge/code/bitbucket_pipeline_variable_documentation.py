from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, _load_params
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper

    Search for a pipeline variable

    :param module: instance of the :class:`AnsibleModule`
    :param bitbucket: instance of the :class:`BitbucketHelper`
    :return: existing variable or None if not found
    :rtype: dict or None

    Return example::

        {
            'name': 'AWS_ACCESS_OBKEY_ID',
            'value': 'x7HU80-a2',
            'type': 'pipeline_variable',
            'secured': False,
            'uuid': '{9ddb0507-439a-495a-99f3-5464f15128127}'
        }

    The `value` key in dict is absent in case of secured variable.
    