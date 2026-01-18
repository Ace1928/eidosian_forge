from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
class ManageIQAlert(object):
    """ Represent a ManageIQ alert. Can be initialized with both the format
    we receive from the server and the format we get from the user.
    """

    def __init__(self, alert):
        self.description = alert['description']
        self.db = alert['db']
        self.enabled = alert['enabled']
        self.options = alert['options']
        self.hash_expression = None
        self.miq_expressipn = None
        if 'hash_expression' in alert:
            self.hash_expression = alert['hash_expression']
        if 'miq_expression' in alert:
            self.miq_expression = alert['miq_expression']
            if 'exp' in self.miq_expression:
                self.miq_expression = self.miq_expression['exp']

    def __eq__(self, other):
        """ Compare two ManageIQAlert objects
        """
        return self.__dict__ == other.__dict__