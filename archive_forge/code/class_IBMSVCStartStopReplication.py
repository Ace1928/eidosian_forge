from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils.basic import AnsibleModule
from traceback import format_exc
class IBMSVCStartStopReplication(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str'), state=dict(type='str', required=True, choices=['started', 'stopped']), force=dict(type='bool', required=False), primary=dict(type='str', choices=['master', 'aux']), clean=dict(type='bool', default=False), access=dict(type='bool', default=False), isgroup=dict(type='bool', default=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.primary = self.module.params.get('primary', None)
        self.clean = self.module.params.get('clean', False)
        self.access = self.module.params.get('access', False)
        self.force = self.module.params.get('force', False)
        self.isgroup = self.module.params.get('isgroup', False)
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def start(self):
        """
        Starts the Metro Mirror or Global Mirror relationship copy process, set
        the direction of copy if undefined, and (optionally) mark the secondary
        volume of the relationship as clean. The relationship must be a
        stand-alone relationship.
        """
        cmdopts = {}
        self.log('self.primary is %s', self.primary)
        if self.primary:
            cmdopts['primary'] = self.primary
        if self.clean:
            cmdopts['clean'] = self.clean
        if self.force:
            cmdopts['force'] = self.force
        if self.isgroup:
            result = self.restapi.svc_run_command(cmd='startrcconsistgrp', cmdopts=cmdopts, cmdargs=[self.name])
            if result == '':
                self.changed = True
                self.log('succeeded to start the remote copy group %s', self.name)
            elif 'message' in result:
                self.changed = True
                self.log('start the remote copy group %s with result message %s', self.name, result['message'])
            else:
                msg = 'Failed to start the remote copy group [%s]' % self.name
                self.module.fail_json(msg=msg)
        else:
            result = self.restapi.svc_run_command(cmd='startrcrelationship', cmdopts=cmdopts, cmdargs=[self.name])
            self.log('start the rcrelationship %s with result %s', self.name, result)
            if result == '':
                self.changed = True
                self.log('succeeded to start the remote copy %s', self.name)
            elif 'message' in result:
                self.changed = True
                self.log('start the rcrelationship %s with result message %s', self.name, result['message'])
            else:
                msg = 'Failed to start the rcrelationship [%s]' % self.name
                self.module.fail_json(msg=msg)

    def stop(self):
        """
        Stops the copy process for a Metro Mirror or Global Mirror stand-alone
        relationship.
        """
        cmdopts = {}
        if self.access:
            cmdopts['access'] = self.access
        if self.isgroup:
            result = self.restapi.svc_run_command(cmd='stoprcconsistgrp', cmdopts=cmdopts, cmdargs=[self.name])
            self.log('stop the remote copy group %s with result %s', self.name, result)
            if result == '':
                self.changed = True
                self.log('succeeded to stop the remote copy group %s', self.name)
            elif 'message' in result:
                self.changed = True
                self.log('stop the remote copy group %s with result message %s', self.name, result['message'])
            else:
                msg = 'Failed to stop the rcrelationship [%s]' % self.name
                self.module.fail_json(msg=msg)
        else:
            result = self.restapi.svc_run_command(cmd='stoprcrelationship', cmdopts=cmdopts, cmdargs=[self.name])
            self.log('stop the rcrelationship %s with result %s', self.name, result)
            if result == '':
                self.changed = True
                self.log('succeeded to stop the remote copy %s', self.name)
            elif 'message' in result:
                self.changed = True
                self.log('stop the rcrelationship %s with result message %s', self.name, result['message'])
            else:
                msg = 'Failed to stop the rcrelationship [%s]' % self.name
                self.module.fail_json(msg=msg)

    def apply(self):
        msg = None
        self.log('self state is %s', self.state)
        if self.module.check_mode:
            msg = 'skipping changes due to check mode.'
        elif self.state == 'started':
            self.start()
            if not self.isgroup:
                msg = 'remote copy [%s] has been started.' % self.name
            else:
                msg = 'remote copy group [%s] has been started.' % self.name
        elif self.state == 'stopped':
            self.stop()
            if not self.isgroup:
                msg = 'remote copy [%s] has been stopped.' % self.name
            else:
                msg = 'remote copy group [%s] has been stopped.' % self.name
        else:
            msg = "Invalid %s state. Supported states are 'started' and 'stopped'" % self.state
        self.module.exit_json(msg=msg, changed=True)