from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class LaunchCtlList(LaunchCtlTask):

    def __init__(self, module, service):
        super(LaunchCtlList, self).__init__(module, service, None)

    def runCommand(self):
        pass