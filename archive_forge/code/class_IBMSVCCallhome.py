from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
class IBMSVCCallhome(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=True, choices=['enabled', 'disabled']), callhome_type=dict(type='str', required=True, choices=['cloud services', 'email', 'both']), proxy_type=dict(type='str', choices=['open_proxy', 'basic_authentication', 'certificate', 'no_proxy']), proxy_url=dict(type='str'), proxy_port=dict(type='int'), proxy_username=dict(type='str'), proxy_password=dict(type='str', no_log=True), sslcert=dict(type='str'), company_name=dict(type='str'), address=dict(type='str'), city=dict(type='str'), province=dict(type='str'), postalcode=dict(type='str'), country=dict(type='str'), location=dict(type='str'), contact_name=dict(type='str'), contact_email=dict(type='str'), phonenumber_primary=dict(type='str'), phonenumber_secondary=dict(type='str'), serverIP=dict(type='str'), serverPort=dict(type='int'), inventory=dict(type='str', choices=['on', 'off']), invemailinterval=dict(type='int'), enhancedcallhome=dict(type='str', choices=['on', 'off']), censorcallhome=dict(type='str', choices=['on', 'off'])))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.state = self.module.params['state']
        self.callhome_type = self.module.params['callhome_type']
        self.company_name = self.module.params['company_name']
        self.address = self.module.params['address']
        self.city = self.module.params['city']
        self.province = self.module.params['province']
        self.postalcode = self.module.params['postalcode']
        self.country = self.module.params['country']
        self.location = self.module.params['location']
        self.contact_name = self.module.params['contact_name']
        self.contact_email = self.module.params['contact_email']
        self.phonenumber_primary = self.module.params['phonenumber_primary']
        self.proxy_type = self.module.params.get('proxy_type', False)
        self.proxy_url = self.module.params.get('proxy_url', False)
        self.proxy_port = self.module.params.get('proxy_port', False)
        self.proxy_username = self.module.params.get('proxy_username', False)
        self.proxy_password = self.module.params.get('proxy_password', False)
        self.sslcert = self.module.params.get('sslcert', False)
        self.phonenumber_secondary = self.module.params.get('phonenumber_secondary', False)
        self.serverIP = self.module.params.get('serverIP', False)
        self.serverPort = self.module.params.get('serverPort', False)
        self.inventory = self.module.params.get('inventory', False)
        self.invemailinterval = self.module.params.get('invemailinterval', False)
        self.enhancedcallhome = self.module.params.get('enhancedcallhome', False)
        self.censorcallhome = self.module.params.get('censorcallhome', False)
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.inventory:
            self.inventory = 'off'
        if not self.enhancedcallhome:
            self.enhancedcallhome = 'off'
        if not self.censorcallhome:
            self.censorcallhome = 'off'
        if self.inventory == 'on':
            if not self.invemailinterval:
                self.module.fail_json(msg='Parameter [invemailinterval] should be configured to use [inventory]')
        if self.invemailinterval:
            if self.inventory == 'off':
                self.module.fail_json(msg="The parameter [inventory] should be configured with 'on' while setting [invemailinterval]")
            if self.invemailinterval not in range(1, 16):
                self.module.fail_json(msg='Parameter [invemailinterval] supported range is 0 to 15')
        if isinstance(self.serverPort, int):
            if self.serverPort not in range(1, 65536):
                self.module.fail_json(msg='Parameter [serverPort] must be in range[1-65535]')
        if isinstance(self.proxy_port, int):
            if self.proxy_port not in range(1, 65536):
                self.module.fail_json(msg='Parameter [proxy_port] must be in range[1-65535]')
        if not self.state:
            self.module.fail_json(msg='Missing mandatory parameter: state')
        if not self.callhome_type:
            self.module.fail_json(msg='Missing mandatory parameter: callhome_type')
        if self.callhome_type in ['email', 'both'] and (not self.serverIP or not self.serverPort) and (self.state == 'enabled'):
            self.module.fail_json(msg='Parameters: serverIP, serverPort are required when callhome_type is email/both')
        if self.state == 'enabled' and self.proxy_type in ['cloud services', 'both'] and self.proxy_type:
            if self.proxy_type == 'open_proxy' and (not self.proxy_url or not self.proxy_port):
                self.module.fail_json(msg='Parameters [proxy_url, proxy_port] required when proxy_type=open_proxy')
            if self.proxy_type == 'basic_authentication' and (not self.proxy_url or not self.proxy_port or (not self.proxy_username) or (not self.proxy_password)):
                self.module.fail_json(msg='Parameters [proxy_url, proxy_port, proxy_username, proxy_password] required when proxy_type=basic_authentication')
            if self.proxy_type == 'certificate' and (not self.proxy_url or not self.proxy_port or (not self.sslcert)):
                self.module.fail_json(msg='Parameters [proxy_url, proxy_port, sslcert] required when proxy_type=certificate')
        if self.state == 'enabled':
            parameters = {'callhome_type': self.callhome_type, 'company_name': self.company_name, 'address': self.address, 'city': self.city, 'province': self.province, 'country': self.country, 'location': self.location, 'contact_name': self.contact_name, 'contact_email': self.contact_email, 'phonenumber_primary': self.phonenumber_primary}
            parameter_not_provided = []
            for parameter in parameters:
                if not parameters[parameter]:
                    parameter_not_provided.append(parameter)
            if parameter_not_provided:
                self.module.fail_json(msg="Parameters {0} are required when state is 'enabled'".format(parameter_not_provided))

    def get_system_data(self):
        return self.restapi.svc_obj_info('lssystem', cmdopts=None, cmdargs=None)

    def probe_system(self, data):
        modify = {}
        if self.invemailinterval:
            if self.invemailinterval != data['inventory_mail_interval']:
                modify['invemailinterval'] = self.invemailinterval
        if self.enhancedcallhome:
            if self.enhancedcallhome != data['enhanced_callhome']:
                modify['enhancedcallhome'] = self.enhancedcallhome
        if self.censorcallhome:
            if self.censorcallhome != data['enhanced_callhome']:
                modify['censorcallhome'] = self.censorcallhome
        return modify

    def update_system(self, modify):
        command = 'chsystem'
        command_options = modify
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Chsystem commands executed.')

    def get_existing_email_user_data(self):
        data = {}
        email_data = self.restapi.svc_obj_info(cmd='lsemailuser', cmdopts=None, cmdargs=None)
        for item in email_data:
            if item['address'] == self.contact_email:
                data = item
        return data

    def check_email_server_exists(self):
        status = False
        data = self.restapi.svc_obj_info(cmd='lsemailserver', cmdopts=None, cmdargs=None)
        for item in data:
            if item['IP_address'] == self.serverIP and int(item['port']) == self.serverPort:
                status = True
                break
        return status

    def check_email_user_exists(self):
        temp = {}
        data = self.restapi.svc_obj_info(cmd='lsemailuser', cmdopts=None, cmdargs=None)
        for item in data:
            if item['address'] == self.contact_email:
                temp = item
                break
        return temp

    def create_email_server(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("Creating email server '%s:%s'.", self.serverIP, self.serverPort)
        command = 'mkemailserver'
        command_options = {'ip': self.serverIP, 'port': self.serverPort}
        cmdargs = None
        result = self.restapi.svc_run_command(command, command_options, cmdargs)
        if 'message' in result:
            self.changed = True
            self.log("create email server result message '%s'", result['message'])
        else:
            self.module.fail_json(msg='Failed to create email server [%s:%s]' % (self.serverIP, self.serverPort))

    def update_email_user(self, data, id):
        command = 'chemailuser'
        command_options = data
        cmdargs = [id]
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Email user updated successfully.')

    def manage_support_email_user(self):
        if self.module.check_mode:
            self.changed = True
            return
        support_email = {}
        selected_email_id = ''
        t = -1 * (time.timezone / 60 / 60)
        if t >= -8 and t <= -4:
            selected_email_id = 'callhome0@de.ibm.com'
        else:
            selected_email_id = 'callhome1@de.ibm.com'
        existing_user = self.restapi.svc_obj_info('lsemailuser', cmdopts=None, cmdargs=None)
        if existing_user:
            for user in existing_user:
                if user['user_type'] == 'support':
                    support_email = user
        if not support_email:
            self.log("Creating support email user '%s'.", selected_email_id)
            command = 'mkemailuser'
            command_options = {'address': selected_email_id, 'usertype': 'support', 'info': 'off', 'warning': 'off'}
            if self.inventory:
                command_options['inventory'] = self.inventory
            cmdargs = None
            result = self.restapi.svc_run_command(command, command_options, cmdargs)
            if 'message' in result:
                self.changed = True
                self.log("create support email user result message '%s'", result['message'])
            else:
                self.module.fail_json(msg='Failed to support create email user [%s]' % self.contact_email)
        else:
            modify = {}
            if support_email['address'] != selected_email_id:
                modify['address'] = selected_email_id
            if self.inventory:
                if support_email['inventory'] != self.inventory:
                    modify['inventory'] = self.inventory
            if modify:
                self.restapi.svc_run_command('chemailuser', modify, [support_email['id']])
                self.log('Updated support user successfully.')

    def create_email_user(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("Creating email user '%s'.", self.contact_email)
        command = 'mkemailuser'
        command_options = {'address': self.contact_email, 'usertype': 'local'}
        if self.inventory:
            command_options['inventory'] = self.inventory
        cmdargs = None
        result = self.restapi.svc_run_command(command, command_options, cmdargs)
        if 'message' in result:
            self.changed = True
            self.log("Create email user result message '%s'.", result['message'])
        else:
            self.module.fail_json(msg='Failed to create email user [%s]' % self.contact_email)

    def enable_email_callhome(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'startemail'
        command_options = {}
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Email callhome enabled.')

    def disable_email_callhome(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'stopemail'
        command_options = {}
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Email callhome disabled.')

    def update_email_data(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'chemail'
        command_options = {}
        if self.contact_email:
            command_options['reply'] = self.contact_email
        if self.contact_name:
            command_options['contact'] = self.contact_name
        if self.phonenumber_primary:
            command_options['primary'] = self.phonenumber_primary
        if self.phonenumber_secondary:
            command_options['alternate'] = self.phonenumber_secondary
        if self.location:
            command_options['location'] = self.location
        if self.company_name:
            command_options['organization'] = self.company_name
        if self.address:
            command_options['address'] = self.address
        if self.city:
            command_options['city'] = self.city
        if self.province:
            command_options['state'] = self.province
        if self.postalcode:
            command_options['zip'] = self.postalcode
        if self.country:
            command_options['country'] = self.country
        cmdargs = None
        if command_options:
            self.restapi.svc_run_command(command, command_options, cmdargs)
            self.log('Email data successfully updated.')

    def get_existing_proxy(self):
        data = {}
        data = self.restapi.svc_obj_info(cmd='lsproxy', cmdopts=None, cmdargs=None)
        return data

    def remove_proxy(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'rmproxy'
        command_options = None
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Proxy removed successfully.')

    def create_proxy(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'mkproxy'
        command_options = {}
        if self.proxy_type == 'open_proxy':
            if self.proxy_url:
                command_options['url'] = self.proxy_url
            if self.proxy_port:
                command_options['port'] = self.proxy_port
        elif self.proxy_type == 'basic_authentication':
            if self.proxy_url:
                command_options['url'] = self.proxy_url
            if self.proxy_port:
                command_options['port'] = self.proxy_port
            if self.proxy_username:
                command_options['username'] = self.proxy_username
            if self.proxy_password:
                command_options['password'] = self.proxy_password
        elif self.proxy_type == 'certificate':
            if self.proxy_url:
                command_options['url'] = self.proxy_url
            if self.proxy_port:
                command_options['port'] = self.proxy_port
            if self.sslcert:
                command_options['sslcert'] = self.sslcert
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Proxy created successfully.')

    def probe_proxy(self, data):
        modify = {}
        if self.proxy_type == 'open_proxy':
            if self.proxy_url:
                if self.proxy_url != data['url']:
                    modify['url'] = self.proxy_url
            if self.proxy_port:
                if int(self.proxy_port) != int(data['port']):
                    modify['port'] = self.proxy_port
        elif self.proxy_type == 'basic_authentication':
            if self.proxy_url:
                if self.proxy_url != data['url']:
                    modify['url'] = self.proxy_url
            if self.proxy_port:
                if self.proxy_port != int(data['port']):
                    modify['port'] = self.proxy_port
            if self.proxy_username:
                if self.proxy_username != data['username']:
                    modify['username'] = self.proxy_username
            if self.proxy_password:
                modify['password'] = self.proxy_password
        elif self.proxy_type == 'certificate':
            if self.proxy_url:
                if self.proxy_url != data['url']:
                    modify['url'] = self.proxy_url
            if self.proxy_port:
                if self.proxy_port != int(data['port']):
                    modify['port'] = self.proxy_port
            if self.sslcert:
                modify['sslcert'] = self.sslcert
        return modify

    def update_proxy(self, data):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'chproxy'
        command_options = data
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Proxy updated successfully.')

    def get_existing_cloud_callhome_data(self):
        data = {}
        command = 'lscloudcallhome'
        command_options = None
        cmdargs = None
        data = self.restapi.svc_obj_info(command, command_options, cmdargs)
        return data

    def enable_cloud_callhome(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'chcloudcallhome'
        command_options = {'enable': True}
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.changed = True
        self.log('Cloud callhome enabled.')

    def test_connection_cloud_callhome(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'sendcloudcallhome'
        command_options = {'connectiontest': True}
        self.restapi.svc_run_command(command, command_options, None)
        self.changed = True
        self.log('Cloud callhome connection tested.')
        time.sleep(3)

    def manage_proxy_server(self):
        proxy_data = self.get_existing_proxy()
        if proxy_data['enabled'] == 'no':
            if self.proxy_type == 'no_proxy':
                self.log('Proxy already disabled.')
            else:
                self.create_proxy()
                self.changed = True
        elif proxy_data['enabled'] == 'yes':
            if self.proxy_type == 'no_proxy':
                self.remove_proxy()
                self.changed = True
            else:
                modify = self.probe_proxy(proxy_data)
                if modify:
                    self.update_proxy(modify)
                    self.changed = True

    def disable_cloud_callhome(self):
        if self.module.check_mode:
            self.changed = True
            return
        command = 'chcloudcallhome'
        command_options = {'disable': True}
        cmdargs = None
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Cloud callhome disabled.')

    def initiate_cloud_callhome(self):
        msg = ''
        attempts = 0
        limit_reached = False
        active_status = False
        self.manage_proxy_server()
        self.update_email_data()
        lsdata = self.get_existing_cloud_callhome_data()
        if lsdata['status'] == 'enabled':
            self.test_connection_cloud_callhome()
        else:
            self.enable_cloud_callhome()
            while not active_status:
                attempts += 1
                if attempts > 10:
                    limit_reached = True
                    break
                lsdata = self.get_existing_cloud_callhome_data()
                if lsdata['status'] == 'enabled':
                    active_status = True
                time.sleep(2)
            if limit_reached:
                msg = 'Callhome with Cloud is enabled. Please check connection to proxy.'
                self.changed = True
                return msg
            if active_status:
                self.test_connection_cloud_callhome()
        msg = 'Callhome with Cloud enabled successfully.'
        self.changed = True
        return msg

    def initiate_email_callhome(self):
        msg = ''
        email_server_exists = self.check_email_server_exists()
        if email_server_exists:
            self.log('Email server already exists.')
        else:
            self.create_email_server()
            self.changed = True
        self.manage_support_email_user()
        email_user_exists = self.check_email_user_exists()
        if email_user_exists:
            email_user_modify = {}
            if email_user_exists['inventory'] != self.inventory:
                email_user_modify['inventory'] = self.inventory
            if email_user_modify:
                self.update_email_user(email_user_modify, email_user_exists['id'])
        else:
            self.create_email_user()
        self.update_email_data()
        self.enable_email_callhome()
        msg = 'Callhome with email enabled successfully.'
        self.changed = True
        return msg

    def apply(self):
        self.changed = False
        msg = None
        self.basic_checks()
        if self.state == 'enabled':
            if self.callhome_type == 'cloud services':
                msg = self.initiate_cloud_callhome()
            elif self.callhome_type == 'email':
                msg = self.initiate_email_callhome()
            elif self.callhome_type == 'both':
                temp_msg = ''
                temp_msg += self.initiate_cloud_callhome()
                temp_msg += ' ' + self.initiate_email_callhome()
                if temp_msg:
                    msg = temp_msg
            system_data = self.get_system_data()
            system_modify = self.probe_system(system_data)
            if system_modify:
                self.update_system(system_modify)
        elif self.state == 'disabled':
            if self.callhome_type == 'cloud services':
                cloud_callhome_data = self.get_existing_cloud_callhome_data()
                if cloud_callhome_data['status'] == 'disabled':
                    msg = 'Callhome with cloud already disabled.'
                elif cloud_callhome_data['status'] == 'enabled':
                    self.disable_cloud_callhome()
                    msg = 'Callhome with cloud disabled successfully.'
                    self.changed = True
            elif self.callhome_type == 'email':
                self.disable_email_callhome()
                msg = 'Callhome with email disabled successfully.'
                self.changed = True
            elif self.callhome_type == 'both':
                self.disable_email_callhome()
                msg = 'Callhome with email disabled successfully.'
                self.changed = True
                cloud_callhome_data = self.get_existing_cloud_callhome_data()
                if cloud_callhome_data['status'] == 'disabled':
                    msg += ' Callhome with cloud already disabled.'
                elif cloud_callhome_data['status'] == 'enabled':
                    self.disable_cloud_callhome()
                    msg += ' Callhome with cloud disabled successfully.'
                    self.changed = True
        self.module.exit_json(msg=msg, changed=self.changed)