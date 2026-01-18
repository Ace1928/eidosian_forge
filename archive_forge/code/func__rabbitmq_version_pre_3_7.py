from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
def _rabbitmq_version_pre_3_7(self, fail_on_error=False):
    """Get the version of the RabbitMQ Server.

        Before version 3.7.6 the `rabbitmqctl` utility did not support the
        `--formatter` flag, so the output has to be parsed using regular expressions.
        """
    version_reg_ex = '{rabbit,\\"RabbitMQ\\",\\"([0-9]+\\.[0-9]+\\.[0-9]+)\\"}'
    rc, output, err = self._exec(['status'], check_rc=False)
    if rc != 0:
        if fail_on_error:
            self.module.fail_json(msg='Could not parse the version of the RabbitMQ server, because `rabbitmqctl status` returned no output.')
        else:
            return None
    reg_ex_res = re.search(version_reg_ex, output, re.IGNORECASE)
    if not reg_ex_res:
        return self._fail(msg='Could not parse the version of the RabbitMQ server from the output of `rabbitmqctl status` command: {output}.'.format(output=output), stop_execution=fail_on_error)
    try:
        return Version.StrictVersion(reg_ex_res.group(1))
    except ValueError as e:
        return self._fail(msg='Could not parse the version of the RabbitMQ server: {exc}.'.format(exc=repr(e)), stop_execution=fail_on_error)