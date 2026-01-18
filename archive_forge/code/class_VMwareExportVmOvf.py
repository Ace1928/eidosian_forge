from __future__ import absolute_import, division, print_function
import os
import hashlib
from time import sleep
from threading import Thread
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VMwareExportVmOvf(PyVmomi):

    def __init__(self, module):
        super(VMwareExportVmOvf, self).__init__(module)
        self.mf_file = ''
        self.ovf_dir = ''
        self.chunk_size = 2 * 2 ** 20
        self.lease_interval = 15
        self.facts = {'device_files': []}
        self.download_timeout = None

    def create_export_dir(self, vm_obj):
        self.ovf_dir = os.path.join(self.params['export_dir'], vm_obj.name)
        if not os.path.exists(self.ovf_dir):
            try:
                os.makedirs(self.ovf_dir)
            except OSError as err:
                self.module.fail_json(msg='Exception caught when create folder %s, with error %s' % (self.ovf_dir, to_text(err)))
        self.mf_file = os.path.join(self.ovf_dir, vm_obj.name + '.mf')

    def download_device_files(self, headers, temp_target_disk, device_url, lease_updater, total_bytes_written, total_bytes_to_write):
        mf_content = 'SHA256(' + os.path.basename(temp_target_disk) + ')= '
        sha256_hash = hashlib.sha256()
        response = None
        with open(self.mf_file, 'a') as mf_handle:
            with open(temp_target_disk, 'wb') as handle:
                try:
                    response = open_url(device_url, headers=headers, validate_certs=False, timeout=self.download_timeout)
                except Exception as err:
                    lease_updater.httpNfcLease.HttpNfcLeaseAbort()
                    lease_updater.stop()
                    self.module.fail_json(msg='Exception caught when getting %s, %s' % (device_url, to_text(err)))
                if not response:
                    lease_updater.httpNfcLease.HttpNfcLeaseAbort()
                    lease_updater.stop()
                    self.module.fail_json(msg='Getting %s failed' % device_url)
                if response.getcode() >= 400:
                    lease_updater.httpNfcLease.HttpNfcLeaseAbort()
                    lease_updater.stop()
                    self.module.fail_json(msg='Getting %s return code %d' % (device_url, response.getcode()))
                current_bytes_written = 0
                block = response.read(self.chunk_size)
                while block:
                    handle.write(block)
                    sha256_hash.update(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                    current_bytes_written += len(block)
                    block = response.read(self.chunk_size)
                written_percent = (current_bytes_written + total_bytes_written) * 100 / total_bytes_to_write
                lease_updater.progressPercent = int(written_percent)
            mf_handle.write(mf_content + sha256_hash.hexdigest() + '\n')
        self.facts['device_files'].append(temp_target_disk)
        return current_bytes_written

    def export_to_ovf_files(self, vm_obj):
        self.create_export_dir(vm_obj=vm_obj)
        export_with_iso = False
        if self.params['export_with_images']:
            export_with_iso = True
        self.download_timeout = self.params['download_timeout']
        ovf_files = []
        http_nfc_lease = vm_obj.ExportVm()
        lease_updater = LeaseProgressUpdater(http_nfc_lease, self.lease_interval)
        total_bytes_written = 0
        total_bytes_to_write = vm_obj.summary.storage.unshared
        if total_bytes_to_write == 0:
            total_bytes_to_write = vm_obj.summary.storage.committed
            if total_bytes_to_write == 0:
                http_nfc_lease.HttpNfcLeaseAbort()
                self.module.fail_json(msg='Total storage space occupied by the VM is 0.')
        headers = {'Accept': 'application/x-vnd.vmware-streamVmdk'}
        cookies = connect.GetStub().cookie
        if cookies:
            headers['Cookie'] = cookies
        lease_updater.start()
        try:
            while True:
                if http_nfc_lease.state == vim.HttpNfcLease.State.ready:
                    for deviceUrl in http_nfc_lease.info.deviceUrl:
                        file_download = False
                        if deviceUrl.targetId and deviceUrl.disk:
                            file_download = True
                        elif deviceUrl.url.split('/')[-1].split('.')[-1] == 'iso':
                            if export_with_iso:
                                file_download = True
                        elif deviceUrl.url.split('/')[-1].split('.')[-1] == 'nvram':
                            if self.host_version_at_least(version=(6, 7, 0), vm_obj=vm_obj):
                                file_download = True
                        else:
                            continue
                        device_file_name = deviceUrl.url.split('/')[-1]
                        if device_file_name.split('.')[0][0:5] == 'disk-':
                            device_file_name = device_file_name.replace('disk', vm_obj.name)
                        temp_target_disk = os.path.join(self.ovf_dir, device_file_name)
                        device_url = deviceUrl.url
                        if '*' in device_url:
                            device_url = device_url.replace('*', self.params['hostname'])
                        if file_download:
                            current_bytes_written = self.download_device_files(headers=headers, temp_target_disk=temp_target_disk, device_url=device_url, lease_updater=lease_updater, total_bytes_written=total_bytes_written, total_bytes_to_write=total_bytes_to_write)
                            total_bytes_written += current_bytes_written
                            ovf_file = vim.OvfManager.OvfFile()
                            ovf_file.deviceId = deviceUrl.key
                            ovf_file.path = device_file_name
                            ovf_file.size = current_bytes_written
                            ovf_files.append(ovf_file)
                    break
                if http_nfc_lease.state == vim.HttpNfcLease.State.initializing:
                    sleep(2)
                    continue
                if http_nfc_lease.state == vim.HttpNfcLease.State.error:
                    lease_updater.stop()
                    self.module.fail_json(msg='Get HTTP NFC lease error %s.' % http_nfc_lease.state.error[0].fault)
            ovf_manager = self.content.ovfManager
            ovf_descriptor_name = vm_obj.name
            ovf_parameters = vim.OvfManager.CreateDescriptorParams()
            ovf_parameters.name = ovf_descriptor_name
            ovf_parameters.ovfFiles = ovf_files
            if self.params['export_with_extraconfig']:
                ovf_parameters.exportOption = ['extraconfig']
            if self.params['export_with_images']:
                ovf_parameters.includeImageFiles = True
            vm_descriptor_result = ovf_manager.CreateDescriptor(obj=vm_obj, cdp=ovf_parameters)
            if vm_descriptor_result.error:
                http_nfc_lease.HttpNfcLeaseAbort()
                lease_updater.stop()
                self.module.fail_json(msg='Create VM descriptor file error %s.' % vm_descriptor_result.error)
            else:
                vm_descriptor = vm_descriptor_result.ovfDescriptor
                ovf_descriptor_path = os.path.join(self.ovf_dir, ovf_descriptor_name + '.ovf')
                sha256_hash = hashlib.sha256()
                with open(self.mf_file, 'a') as mf_handle:
                    with open(ovf_descriptor_path, 'w') as handle:
                        handle.write(vm_descriptor)
                        sha256_hash.update(to_bytes(vm_descriptor))
                    mf_handle.write('SHA256(' + os.path.basename(ovf_descriptor_path) + ')= ' + sha256_hash.hexdigest() + '\n')
                http_nfc_lease.HttpNfcLeaseProgress(100)
                http_nfc_lease.HttpNfcLeaseComplete()
                lease_updater.stop()
                self.facts.update({'manifest': self.mf_file, 'ovf_file': ovf_descriptor_path})
        except Exception as err:
            kwargs = {'changed': False, 'failed': True, 'msg': 'get exception: %s' % to_text(err)}
            http_nfc_lease.HttpNfcLeaseAbort()
            lease_updater.stop()
            return kwargs
        return {'changed': True, 'failed': False, 'instance': self.facts}