from __future__ import absolute_import, division, print_function
import os
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils._text import to_native, to_text, to_bytes
class NetAppESeriesDriveFirmware(NetAppESeriesModule):
    WAIT_TIMEOUT_SEC = 60 * 15

    def __init__(self):
        ansible_options = dict(firmware=dict(type='list', required=True), wait_for_completion=dict(type='bool', default=False), ignore_inaccessible_drives=dict(type='bool', default=False), upgrade_drives_online=dict(type='bool', default=True))
        super(NetAppESeriesDriveFirmware, self).__init__(ansible_options=ansible_options, web_services_version='02.00.0000.0000', supports_check_mode=True)
        args = self.module.params
        self.firmware_list = args['firmware']
        self.wait_for_completion = args['wait_for_completion']
        self.ignore_inaccessible_drives = args['ignore_inaccessible_drives']
        self.upgrade_drives_online = args['upgrade_drives_online']
        self.upgrade_list_cache = None
        self.upgrade_required_cache = None
        self.upgrade_in_progress = False
        self.drive_info_cache = None

    def upload_firmware(self):
        """Ensure firmware has been upload prior to uploaded."""
        for firmware in self.firmware_list:
            firmware_name = os.path.basename(firmware)
            files = [('file', firmware_name, firmware)]
            headers, data = create_multipart_formdata(files)
            try:
                rc, response = self.request('/files/drive', method='POST', headers=headers, data=data)
            except Exception as error:
                self.module.fail_json(msg='Failed to upload drive firmware [%s]. Array [%s]. Error [%s].' % (firmware_name, self.ssid, to_native(error)))

    def upgrade_list(self):
        """Determine whether firmware is compatible with the specified drives."""
        if self.upgrade_list_cache is None:
            self.upgrade_list_cache = list()
            try:
                rc, response = self.request('storage-systems/%s/firmware/drives' % self.ssid)
                for firmware in self.firmware_list:
                    filename = os.path.basename(firmware)
                    for uploaded_firmware in response['compatibilities']:
                        if uploaded_firmware['filename'] == filename:
                            drive_reference_list = []
                            for drive in uploaded_firmware['compatibleDrives']:
                                try:
                                    rc, drive_info = self.request('storage-systems/%s/drives/%s' % (self.ssid, drive['driveRef']))
                                    if drive_info['firmwareVersion'] != uploaded_firmware['firmwareVersion'] and uploaded_firmware['firmwareVersion'] in uploaded_firmware['supportedFirmwareVersions']:
                                        if self.ignore_inaccessible_drives or (not drive_info['offline'] and drive_info['available']):
                                            drive_reference_list.append(drive['driveRef'])
                                        if not drive['onlineUpgradeCapable'] and self.upgrade_drives_online:
                                            self.module.fail_json(msg='Drive is not capable of online upgrade. Array [%s]. Drive [%s].' % (self.ssid, drive['driveRef']))
                                except Exception as error:
                                    self.module.fail_json(msg='Failed to retrieve drive information. Array [%s]. Drive [%s]. Error [%s].' % (self.ssid, drive['driveRef'], to_native(error)))
                            if drive_reference_list:
                                self.upgrade_list_cache.extend([{'filename': filename, 'driveRefList': drive_reference_list}])
            except Exception as error:
                self.module.fail_json(msg='Failed to complete compatibility and health check. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        return self.upgrade_list_cache

    def wait_for_upgrade_completion(self):
        """Wait for drive firmware upgrade to complete."""
        drive_references = [reference for drive in self.upgrade_list() for reference in drive['driveRefList']]
        last_status = None
        for attempt in range(int(self.WAIT_TIMEOUT_SEC / 5)):
            try:
                rc, response = self.request('storage-systems/%s/firmware/drives/state' % self.ssid)
                for status in response['driveStatus']:
                    last_status = status
                    if status['driveRef'] in drive_references:
                        if status['status'] == 'okay':
                            continue
                        elif status['status'] in ['inProgress', 'inProgressRecon', 'pending', 'notAttempted']:
                            break
                        else:
                            self.module.fail_json(msg='Drive firmware upgrade failed. Array [%s]. Drive [%s]. Status [%s].' % (self.ssid, status['driveRef'], status['status']))
                else:
                    self.upgrade_in_progress = False
                    break
            except Exception as error:
                self.module.fail_json(msg='Failed to retrieve drive status. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
            sleep(5)
        else:
            self.module.fail_json(msg='Timed out waiting for drive firmware upgrade. Array [%s]. Status [%s].' % (self.ssid, last_status))

    def upgrade(self):
        """Apply firmware to applicable drives."""
        try:
            rc, response = self.request('storage-systems/%s/firmware/drives/initiate-upgrade?onlineUpdate=%s' % (self.ssid, 'true' if self.upgrade_drives_online else 'false'), method='POST', data=self.upgrade_list())
            self.upgrade_in_progress = True
        except Exception as error:
            self.module.fail_json(msg='Failed to upgrade drive firmware. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        if self.wait_for_completion:
            self.wait_for_upgrade_completion()

    def apply(self):
        """Apply firmware policy has been enforced on E-Series storage system."""
        self.upload_firmware()
        if self.upgrade_list() and (not self.module.check_mode):
            self.upgrade()
        self.module.exit_json(changed=True if self.upgrade_list() else False, upgrade_in_process=self.upgrade_in_progress)