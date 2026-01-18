from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class ProxmoxTemplateAnsible(ProxmoxAnsible):

    def get_template(self, node, storage, content_type, template):
        try:
            return [True for tmpl in self.proxmox_api.nodes(node).storage(storage).content.get() if tmpl['volid'] == '%s:%s/%s' % (storage, content_type, template)]
        except Exception as e:
            self.module.fail_json(msg="Failed to retrieve template '%s:%s/%s': %s" % (storage, content_type, template, e))

    def task_status(self, node, taskid, timeout):
        """
        Check the task status and wait until the task is completed or the timeout is reached.
        """
        while timeout:
            if self.api_task_ok(node, taskid):
                return True
            timeout = timeout - 1
            if timeout == 0:
                self.module.fail_json(msg='Reached timeout while waiting for uploading/downloading template. Last line in task before timeout: %s' % self.proxmox_api.node(node).tasks(taskid).log.get()[:1])
            time.sleep(1)
        return False

    def upload_template(self, node, storage, content_type, realpath, timeout):
        stats = os.stat(realpath)
        if LooseVersion(self.proxmoxer_version) >= LooseVersion('1.2.0') and stats.st_size > 268435456 and (not HAS_REQUESTS_TOOLBELT):
            self.module.fail_json(msg="'requests_toolbelt' module is required to upload files larger than 256MB", exception=missing_required_lib('requests_toolbelt'))
        try:
            taskid = self.proxmox_api.nodes(node).storage(storage).upload.post(content=content_type, filename=open(realpath, 'rb'))
            return self.task_status(node, taskid, timeout)
        except Exception as e:
            self.module.fail_json(msg='Uploading template %s failed with error: %s' % (realpath, e))

    def download_template(self, node, storage, template, timeout):
        try:
            taskid = self.proxmox_api.nodes(node).aplinfo.post(storage=storage, template=template)
            return self.task_status(node, taskid, timeout)
        except Exception as e:
            self.module.fail_json(msg='Downloading template %s failed with error: %s' % (template, e))

    def delete_template(self, node, storage, content_type, template, timeout):
        volid = '%s:%s/%s' % (storage, content_type, template)
        self.proxmox_api.nodes(node).storage(storage).content.delete(volid)
        while timeout:
            if not self.get_template(node, storage, content_type, template):
                return True
            timeout = timeout - 1
            if timeout == 0:
                self.module.fail_json(msg='Reached timeout while waiting for deleting template.')
            time.sleep(1)
        return False