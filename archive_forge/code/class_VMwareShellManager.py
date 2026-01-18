from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareShellManager(PyVmomi):

    def __init__(self, module):
        super(VMwareShellManager, self).__init__(module)
        datacenter_name = module.params['datacenter']
        cluster_name = module.params['cluster']
        folder = module.params['folder']
        try:
            self.pm = self.content.guestOperationsManager.processManager
        except vmodl.fault.ManagedObjectNotFound:
            pass
        self.timeout = self.params.get('timeout', 3600)
        self.wait_for_pid = self.params.get('wait_for_process', False)
        datacenter = None
        if datacenter_name:
            datacenter = find_datacenter_by_name(self.content, datacenter_name)
            if not datacenter:
                module.fail_json(changed=False, msg='Unable to find %(datacenter)s datacenter' % module.params)
        cluster = None
        if cluster_name:
            cluster = find_cluster_by_name(self.content, cluster_name, datacenter)
            if not cluster:
                module.fail_json(changed=False, msg='Unable to find %(cluster)s cluster' % module.params)
        if module.params['vm_id_type'] == 'inventory_path':
            vm = find_vm_by_id(self.content, vm_id=module.params['vm_id'], vm_id_type='inventory_path', folder=folder)
        else:
            vm = find_vm_by_id(self.content, vm_id=module.params['vm_id'], vm_id_type=module.params['vm_id_type'], datacenter=datacenter, cluster=cluster)
        if not vm:
            module.fail_json(msg='Unable to find virtual machine.')
        tools_status = vm.guest.toolsStatus
        if tools_status in ['toolsNotInstalled', 'toolsNotRunning']:
            self.module.fail_json(msg='VMwareTools is not installed or is not running in the guest. VMware Tools are necessary to run this module.')
        try:
            self.execute_command(vm, module.params)
        except vmodl.RuntimeFault as runtime_fault:
            module.fail_json(changed=False, msg=to_native(runtime_fault.msg))
        except vmodl.MethodFault as method_fault:
            module.fail_json(changed=False, msg=to_native(method_fault.msg))
        except Exception as e:
            module.fail_json(changed=False, msg=to_native(e))

    def execute_command(self, vm, params):
        vm_username = params['vm_username']
        vm_password = params['vm_password']
        program_path = params['vm_shell']
        args = params['vm_shell_args']
        env = params['vm_shell_env']
        cwd = params['vm_shell_cwd']
        credentials = vim.vm.guest.NamePasswordAuthentication(username=vm_username, password=vm_password)
        cmd_spec = vim.vm.guest.ProcessManager.ProgramSpec(arguments=args, envVariables=env, programPath=program_path, workingDirectory=cwd)
        res = self.pm.StartProgramInGuest(vm=vm, auth=credentials, spec=cmd_spec)
        if self.wait_for_pid:
            res_data = self.wait_for_process(vm, res, credentials)
            results = dict(uuid=vm.summary.config.uuid, owner=res_data.owner, start_time=res_data.startTime.isoformat(), end_time=res_data.endTime.isoformat(), exit_code=res_data.exitCode, name=res_data.name, cmd_line=res_data.cmdLine)
            if res_data.exitCode != 0:
                results['msg'] = 'Failed to execute command'
                results['changed'] = False
                results['failed'] = True
                self.module.fail_json(**results)
            else:
                results['changed'] = True
                results['failed'] = False
                self.module.exit_json(**results)
        else:
            self.module.exit_json(changed=True, uuid=vm.summary.config.uuid, msg=res)

    def process_exists_in_guest(self, vm, pid, creds):
        res = self.pm.ListProcessesInGuest(vm, creds, pids=[pid])
        if not res:
            self.module.fail_json(changed=False, msg='ListProcessesInGuest: None (unexpected)')
        res = res[0]
        if res.exitCode is None:
            return (True, None)
        else:
            return (False, res)

    def wait_for_process(self, vm, pid, creds):
        start_time = time.time()
        while True:
            current_time = time.time()
            process_status, res_data = self.process_exists_in_guest(vm, pid, creds)
            if not process_status:
                return res_data
            elif current_time - start_time >= self.timeout:
                self.module.fail_json(msg='Timeout waiting for process to complete.', vm=vm._moId, pid=pid, start_time=start_time, current_time=current_time, timeout=self.timeout)
            else:
                time.sleep(5)