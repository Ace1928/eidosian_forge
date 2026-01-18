from __future__ import absolute_import, division, print_function
import os
import hashlib
from time import sleep
from threading import Thread
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class LeaseProgressUpdater(Thread):

    def __init__(self, http_nfc_lease, update_interval):
        Thread.__init__(self)
        self._running = True
        self.httpNfcLease = http_nfc_lease
        self.updateInterval = update_interval
        self.progressPercent = 0

    def set_progress_percent(self, progress_percent):
        self.progressPercent = progress_percent

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                if self.httpNfcLease.state == vim.HttpNfcLease.State.done:
                    return
                self.httpNfcLease.HttpNfcLeaseProgress(self.progressPercent)
                sleep_sec = 0
                while True:
                    if self.httpNfcLease.state == vim.HttpNfcLease.State.done or self.httpNfcLease.state == vim.HttpNfcLease.State.error:
                        return
                    sleep_sec += 1
                    sleep(1)
                    if sleep_sec == self.updateInterval:
                        break
            except Exception:
                return