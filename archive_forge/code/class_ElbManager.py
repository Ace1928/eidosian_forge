from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
class ElbManager:
    """Handles ELB creation and destruction"""

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.listeners = module.params['listeners']
        self.purge_listeners = module.params['purge_listeners']
        self.instance_ids = module.params['instance_ids']
        self.purge_instance_ids = module.params['purge_instance_ids']
        self.zones = module.params['zones']
        self.purge_zones = module.params['purge_zones']
        self.health_check = module.params['health_check']
        self.access_logs = module.params['access_logs']
        self.subnets = module.params['subnets']
        self.purge_subnets = module.params['purge_subnets']
        self.scheme = module.params['scheme']
        self.connection_draining_timeout = module.params['connection_draining_timeout']
        self.idle_timeout = module.params['idle_timeout']
        self.cross_az_load_balancing = module.params['cross_az_load_balancing']
        self.stickiness = module.params['stickiness']
        self.wait = module.params['wait']
        self.wait_timeout = module.params['wait_timeout']
        self.tags = module.params['tags']
        self.purge_tags = module.params['purge_tags']
        self.changed = False
        self.status = 'gone'
        retry_decorator = AWSRetry.jittered_backoff()
        self.client = self.module.client('elb', retry_decorator=retry_decorator)
        self.ec2_client = self.module.client('ec2', retry_decorator=retry_decorator)
        security_group_names = module.params['security_group_names']
        self.security_group_ids = module.params['security_group_ids']
        self._update_descriptions()
        if security_group_names:
            if self.elb and self.elb.get('Subnets', None):
                subnets = set(self.elb.get('Subnets') + list(self.subnets or []))
            else:
                subnets = set(self.subnets)
            if subnets:
                vpc_id = self._get_vpc_from_subnets(subnets)
            else:
                vpc_id = None
            try:
                self.security_group_ids = self._get_ec2_security_group_ids_from_names(sec_group_list=security_group_names, vpc_id=vpc_id)
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                module.fail_json_aws(e, msg='Failed to convert security group names to IDs, try using security group IDs rather than names')

    def _update_descriptions(self):
        try:
            self.elb = self._get_elb()
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Unable to describe load balancer')
        try:
            self.elb_attributes = self._get_elb_attributes()
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Unable to describe load balancer attributes')
        try:
            self.elb_policies = self._get_elb_policies()
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Unable to describe load balancer policies')
        try:
            self.elb_health = self._get_elb_instance_health()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Unable to describe load balancer instance health')

    def validate_params(self, state=None):
        problem_found = False
        problem_found |= self._validate_listeners(self.listeners)
        problem_found |= self._validate_health_check(self.health_check)
        problem_found |= self._validate_stickiness(self.stickiness)
        if state == 'present':
            problem_found |= self._validate_creation_requirements()
        problem_found |= self._validate_access_logs(self.access_logs)

    @property
    def check_mode(self):
        return self.module.check_mode

    def _get_elb_policies(self):
        try:
            attributes = self.client.describe_load_balancer_policies(LoadBalancerName=self.name)
        except is_boto3_error_code(['LoadBalancerNotFound', 'LoadBalancerAttributeNotFoundException']):
            return {}
        except is_boto3_error_code('AccessDenied'):
            self.module.warn('Access Denied trying to describe load balancer policies')
            return {}
        return attributes['PolicyDescriptions']

    def _get_elb_instance_health(self):
        try:
            instance_health = self.client.describe_instance_health(LoadBalancerName=self.name)
        except is_boto3_error_code(['LoadBalancerNotFound', 'LoadBalancerAttributeNotFoundException']):
            return []
        except is_boto3_error_code('AccessDenied'):
            self.module.warn('Access Denied trying to describe instance health')
            return []
        return instance_health['InstanceStates']

    def _get_elb_attributes(self):
        try:
            attributes = self.client.describe_load_balancer_attributes(LoadBalancerName=self.name)
        except is_boto3_error_code(['LoadBalancerNotFound', 'LoadBalancerAttributeNotFoundException']):
            return {}
        except is_boto3_error_code('AccessDenied'):
            self.module.warn('Access Denied trying to describe load balancer attributes')
            return {}
        return attributes['LoadBalancerAttributes']

    def _get_elb(self):
        try:
            elbs = self._describe_loadbalancer(self.name)
        except is_boto3_error_code('LoadBalancerNotFound'):
            return None
        if len(elbs) > 1:
            self.module.fail_json(f'Found multiple ELBs with name {self.name}')
        self.status = 'exists' if self.status == 'gone' else self.status
        return elbs[0]

    def _delete_elb(self):
        try:
            if not self.check_mode:
                self.client.delete_load_balancer(aws_retry=True, LoadBalancerName=self.name)
            self.changed = True
            self.status = 'deleted'
        except is_boto3_error_code('LoadBalancerNotFound'):
            return False
        return True

    def _create_elb(self):
        listeners = list((self._format_listener(l) for l in self.listeners))
        if not self.scheme:
            self.scheme = 'internet-facing'
        params = dict(LoadBalancerName=self.name, AvailabilityZones=self.zones, SecurityGroups=self.security_group_ids, Subnets=self.subnets, Listeners=listeners, Scheme=self.scheme)
        params = scrub_none_parameters(params)
        if self.tags:
            params['Tags'] = ansible_dict_to_boto3_tag_list(self.tags)
        if not self.check_mode:
            self.client.create_load_balancer(aws_retry=True, **params)
            self.elb = self._get_elb()
        self.changed = True
        self.status = 'created'
        return True

    def _format_listener(self, listener, inject_protocol=False):
        """Formats listener into the format needed by the
        ELB API"""
        listener = scrub_none_parameters(listener)
        for protocol in ['protocol', 'instance_protocol']:
            if protocol in listener:
                listener[protocol] = listener[protocol].upper()
        if inject_protocol and 'instance_protocol' not in listener:
            listener['instance_protocol'] = listener['protocol']
        listener.pop('proxy_protocol', None)
        ssl_id = listener.pop('ssl_certificate_id', None)
        formatted_listener = snake_dict_to_camel_dict(listener, True)
        if ssl_id:
            formatted_listener['SSLCertificateId'] = ssl_id
        return formatted_listener

    def _format_healthcheck_target(self):
        """Compose target string from healthcheck parameters"""
        protocol = self.health_check['ping_protocol'].upper()
        path = ''
        if protocol in ['HTTP', 'HTTPS'] and 'ping_path' in self.health_check:
            path = self.health_check['ping_path']
        return f'{protocol}:{self.health_check['ping_port']}{path}'

    def _format_healthcheck(self):
        return dict(Target=self._format_healthcheck_target(), Timeout=self.health_check['timeout'], Interval=self.health_check['interval'], UnhealthyThreshold=self.health_check['unhealthy_threshold'], HealthyThreshold=self.health_check['healthy_threshold'])

    def ensure_ok(self):
        """Create the ELB"""
        if not self.elb:
            try:
                self._create_elb()
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                self.module.fail_json_aws(e, msg='Failed to create load balancer')
            try:
                self.elb_attributes = self._get_elb_attributes()
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg='Unable to describe load balancer attributes')
            self._wait_created()
        elif self._check_scheme():
            self.ensure_gone()
            self._wait_gone(True)
            try:
                self._create_elb()
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                self.module.fail_json_aws(e, msg='Failed to recreate load balancer')
            try:
                self.elb_attributes = self._get_elb_attributes()
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg='Unable to describe load balancer attributes')
        else:
            self._set_subnets()
            self._set_zones()
            self._set_security_groups()
            self._set_elb_listeners()
            self._set_tags()
        self._set_health_check()
        self._set_elb_attributes()
        self._set_backend_policies()
        self._set_stickiness_policies()
        self._set_instance_ids()

    def ensure_gone(self):
        """Destroy the ELB"""
        if self.elb:
            try:
                self._delete_elb()
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                self.module.fail_json_aws(e, msg='Failed to delete load balancer')
        self._wait_gone()

    def _wait_gone(self, wait=None):
        if not wait and (not self.wait):
            return
        try:
            self._wait_for_elb_removed()
            self._wait_for_elb_interface_removed()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed while waiting for load balancer deletion')

    def _wait_created(self, wait=False):
        if not wait and (not self.wait):
            return
        try:
            self._wait_for_elb_created()
            self._wait_for_elb_interface_created()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed while waiting for load balancer deletion')

    def get_load_balancer(self):
        self._update_descriptions()
        elb = dict(self.elb or {})
        if not elb:
            return {}
        elb['LoadBalancerAttributes'] = self.elb_attributes
        elb['LoadBalancerPolicies'] = self.elb_policies
        load_balancer = camel_dict_to_snake_dict(elb)
        try:
            load_balancer['tags'] = self._get_tags()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to get load balancer tags')
        return load_balancer

    def get_info(self):
        self._update_descriptions()
        if not self.elb:
            return dict(name=self.name, status=self.status, region=self.module.region)
        check_elb = dict(self.elb)
        check_elb_attrs = dict(self.elb_attributes or {})
        check_policies = check_elb.get('Policies', {})
        try:
            lb_cookie_policy = check_policies['LBCookieStickinessPolicies'][0]['PolicyName']
        except (KeyError, IndexError):
            lb_cookie_policy = None
        try:
            app_cookie_policy = check_policies['AppCookieStickinessPolicies'][0]['PolicyName']
        except (KeyError, IndexError):
            app_cookie_policy = None
        health_check = camel_dict_to_snake_dict(check_elb.get('HealthCheck', {}))
        backend_policies = list()
        for port, policies in self._get_backend_policies().items():
            for policy in policies:
                backend_policies.append(f'{port}:{policy}')
        info = dict(name=check_elb.get('LoadBalancerName'), dns_name=check_elb.get('DNSName'), zones=check_elb.get('AvailabilityZones'), security_group_ids=check_elb.get('SecurityGroups'), status=self.status, subnets=check_elb.get('Subnets'), scheme=check_elb.get('Scheme'), hosted_zone_name=check_elb.get('CanonicalHostedZoneName'), hosted_zone_id=check_elb.get('CanonicalHostedZoneNameID'), lb_cookie_policy=lb_cookie_policy, app_cookie_policy=app_cookie_policy, proxy_policy=self._get_proxy_protocol_policy(), backends=backend_policies, instances=self._get_instance_ids(), out_of_service_count=0, in_service_count=0, unknown_instance_state_count=0, region=self.module.region, health_check=health_check)
        instance_health = camel_dict_to_snake_dict(dict(InstanceHealth=self.elb_health))
        info.update(instance_health)
        if info['instance_health']:
            for instance_state in info['instance_health']:
                if instance_state['state'] == 'InService':
                    info['in_service_count'] += 1
                elif instance_state['state'] == 'OutOfService':
                    info['out_of_service_count'] += 1
                else:
                    info['unknown_instance_state_count'] += 1
        listeners = check_elb.get('ListenerDescriptions', [])
        if listeners:
            info['listeners'] = list((self._api_listener_as_tuple(l['Listener']) for l in listeners))
        else:
            info['listeners'] = []
        try:
            info['connection_draining_timeout'] = check_elb_attrs['ConnectionDraining']['Timeout']
        except KeyError:
            pass
        try:
            info['idle_timeout'] = check_elb_attrs['ConnectionSettings']['IdleTimeout']
        except KeyError:
            pass
        try:
            is_enabled = check_elb_attrs['CrossZoneLoadBalancing']['Enabled']
            info['cross_az_load_balancing'] = 'yes' if is_enabled else 'no'
        except KeyError:
            pass
        try:
            info['tags'] = self._get_tags()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to get load balancer tags')
        return info

    @property
    def _waiter_config(self):
        delay = min(10, self.wait_timeout)
        max_attempts = self.wait_timeout // delay
        return {'Delay': delay, 'MaxAttempts': max_attempts}

    def _wait_for_elb_created(self):
        if self.check_mode:
            return True
        waiter = get_waiter(self.client, 'load_balancer_created')
        try:
            waiter.wait(WaiterConfig=self._waiter_config, LoadBalancerNames=[self.name])
        except botocore.exceptions.WaiterError as e:
            self.module.fail_json_aws(e, 'Timeout waiting for ELB removal')
        return True

    def _wait_for_elb_interface_created(self):
        if self.check_mode:
            return True
        waiter = get_waiter(self.ec2_client, 'network_interface_available')
        filters = ansible_dict_to_boto3_filter_list({'requester-id': 'amazon-elb', 'description': f'ELB {self.name}'})
        try:
            waiter.wait(WaiterConfig=self._waiter_config, Filters=filters)
        except botocore.exceptions.WaiterError as e:
            self.module.fail_json_aws(e, 'Timeout waiting for ELB Interface removal')
        return True

    def _wait_for_elb_removed(self):
        if self.check_mode:
            return True
        waiter = get_waiter(self.client, 'load_balancer_deleted')
        try:
            waiter.wait(WaiterConfig=self._waiter_config, LoadBalancerNames=[self.name])
        except botocore.exceptions.WaiterError as e:
            self.module.fail_json_aws(e, 'Timeout waiting for ELB removal')
        return True

    def _wait_for_elb_interface_removed(self):
        if self.check_mode:
            return True
        waiter = get_waiter(self.ec2_client, 'network_interface_deleted')
        filters = ansible_dict_to_boto3_filter_list({'requester-id': 'amazon-elb', 'description': f'ELB {self.name}'})
        try:
            waiter.wait(WaiterConfig=self._waiter_config, Filters=filters)
        except botocore.exceptions.WaiterError as e:
            self.module.fail_json_aws(e, 'Timeout waiting for ELB Interface removal')
        return True

    def _wait_for_instance_state(self, waiter_name, instances):
        if not instances:
            return False
        if self.check_mode:
            return True
        waiter = get_waiter(self.client, waiter_name)
        instance_list = list((dict(InstanceId=instance) for instance in instances))
        try:
            waiter.wait(WaiterConfig=self._waiter_config, LoadBalancerName=self.name, Instances=instance_list)
        except botocore.exceptions.WaiterError as e:
            self.module.fail_json_aws(e, 'Timeout waiting for ELB Instance State')
        return True

    def _create_elb_listeners(self, listeners):
        """Takes a list of listener definitions and creates them"""
        if not listeners:
            return False
        self.changed = True
        if self.check_mode:
            return True
        self.client.create_load_balancer_listeners(aws_retry=True, LoadBalancerName=self.name, Listeners=listeners)
        return True

    def _delete_elb_listeners(self, ports):
        """Takes a list of listener ports and deletes them from the ELB"""
        if not ports:
            return False
        self.changed = True
        if self.check_mode:
            return True
        self.client.delete_load_balancer_listeners(aws_retry=True, LoadBalancerName=self.name, LoadBalancerPorts=ports)
        return True

    def _set_elb_listeners(self):
        """
        Creates listeners specified by self.listeners; overwrites existing
        listeners on these ports; removes extraneous listeners
        """
        if not self.listeners:
            return False
        new_listeners = list((self._format_listener(l, True) for l in self.listeners))
        existing_listeners = list((l['Listener'] for l in self.elb['ListenerDescriptions']))
        listeners_to_remove = list((l for l in existing_listeners if l not in new_listeners))
        listeners_to_add = list((l for l in new_listeners if l not in existing_listeners))
        changed = False
        if self.purge_listeners:
            ports_to_remove = list((l['LoadBalancerPort'] for l in listeners_to_remove))
        else:
            old_ports = set((l['LoadBalancerPort'] for l in listeners_to_remove))
            new_ports = set((l['LoadBalancerPort'] for l in listeners_to_add))
            ports_to_remove = list(old_ports & new_ports)
        try:
            changed |= self._delete_elb_listeners(ports_to_remove)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to remove listeners from load balancer')
        try:
            changed |= self._create_elb_listeners(listeners_to_add)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to remove listeners from load balancer')
        return changed

    def _api_listener_as_tuple(self, listener):
        """Adds ssl_certificate_id to ELB API tuple if present"""
        base_tuple = [listener.get('LoadBalancerPort'), listener.get('InstancePort'), listener.get('Protocol'), listener.get('InstanceProtocol')]
        if listener.get('SSLCertificateId', False):
            base_tuple.append(listener.get('SSLCertificateId'))
        return tuple(base_tuple)

    def _attach_subnets(self, subnets):
        if not subnets:
            return False
        self.changed = True
        if self.check_mode:
            return True
        self.client.attach_load_balancer_to_subnets(aws_retry=True, LoadBalancerName=self.name, Subnets=subnets)
        return True

    def _detach_subnets(self, subnets):
        if not subnets:
            return False
        self.changed = True
        if self.check_mode:
            return True
        self.client.detach_load_balancer_from_subnets(aws_retry=True, LoadBalancerName=self.name, Subnets=subnets)
        return True

    def _set_subnets(self):
        """Determine which subnets need to be attached or detached on the ELB"""
        if self.subnets is None:
            return False
        changed = False
        if self.purge_subnets:
            subnets_to_detach = list(set(self.elb['Subnets']) - set(self.subnets))
        else:
            subnets_to_detach = list()
        subnets_to_attach = list(set(self.subnets) - set(self.elb['Subnets']))
        try:
            changed |= self._detach_subnets(subnets_to_detach)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to detach subnets from load balancer')
        try:
            changed |= self._attach_subnets(subnets_to_attach)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to attach subnets to load balancer')
        return changed

    def _check_scheme(self):
        """Determine if the current scheme is different than the scheme of the ELB"""
        if self.scheme:
            if self.elb['Scheme'] != self.scheme:
                return True
        return False

    def _enable_zones(self, zones):
        if not zones:
            return False
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.enable_availability_zones_for_load_balancer(aws_retry=True, LoadBalancerName=self.name, AvailabilityZones=zones)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to enable zones for load balancer')
        return True

    def _disable_zones(self, zones):
        if not zones:
            return False
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.disable_availability_zones_for_load_balancer(aws_retry=True, LoadBalancerName=self.name, AvailabilityZones=zones)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to disable zones for load balancer')
        return True

    def _set_zones(self):
        """Determine which zones need to be enabled or disabled on the ELB"""
        if self.zones is None:
            return False
        changed = False
        if self.purge_zones:
            zones_to_disable = list(set(self.elb['AvailabilityZones']) - set(self.zones))
        else:
            zones_to_disable = list()
        zones_to_enable = list(set(self.zones) - set(self.elb['AvailabilityZones']))
        try:
            changed |= self._enable_zones(zones_to_enable)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to enable zone on load balancer')
        try:
            changed |= self._disable_zones(zones_to_disable)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to attach zone to load balancer')
        return changed

    def _set_security_groups(self):
        if not self.security_group_ids:
            return False
        if set(self.elb['SecurityGroups']) == set(self.security_group_ids):
            return False
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.apply_security_groups_to_load_balancer(aws_retry=True, LoadBalancerName=self.name, SecurityGroups=self.security_group_ids)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to apply security groups to load balancer')
        return True

    def _set_health_check(self):
        if not self.health_check:
            return False
        'Set health check values on ELB as needed'
        health_check_config = self._format_healthcheck()
        if self.elb and health_check_config == self.elb['HealthCheck']:
            return False
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.configure_health_check(aws_retry=True, LoadBalancerName=self.name, HealthCheck=health_check_config)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to apply healthcheck to load balancer')
        return True

    def _set_elb_attributes(self):
        attributes = {}
        if self.cross_az_load_balancing is not None:
            attr = dict(Enabled=self.cross_az_load_balancing)
            if not self.elb_attributes.get('CrossZoneLoadBalancing', None) == attr:
                attributes['CrossZoneLoadBalancing'] = attr
        if self.idle_timeout is not None:
            attr = dict(IdleTimeout=self.idle_timeout)
            if not self.elb_attributes.get('ConnectionSettings', None) == attr:
                attributes['ConnectionSettings'] = attr
        if self.connection_draining_timeout is not None:
            curr_attr = dict(self.elb_attributes.get('ConnectionDraining', {}))
            if self.connection_draining_timeout == 0:
                attr = dict(Enabled=False)
                curr_attr.pop('Timeout', None)
            else:
                attr = dict(Enabled=True, Timeout=self.connection_draining_timeout)
            if not curr_attr == attr:
                attributes['ConnectionDraining'] = attr
        if self.access_logs is not None:
            curr_attr = dict(self.elb_attributes.get('AccessLog', {}))
            if not self.access_logs.get('enabled'):
                curr_attr = dict(Enabled=curr_attr.get('Enabled', False))
                attr = dict(Enabled=self.access_logs.get('enabled'))
            else:
                attr = dict(Enabled=True, S3BucketName=self.access_logs['s3_location'], S3BucketPrefix=self.access_logs.get('s3_prefix', ''), EmitInterval=self.access_logs.get('interval', 60))
            if not curr_attr == attr:
                attributes['AccessLog'] = attr
        if not attributes:
            return False
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.modify_load_balancer_attributes(aws_retry=True, LoadBalancerName=self.name, LoadBalancerAttributes=attributes)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to apply load balancer attrbutes')

    def _proxy_policy_name(self):
        return 'ProxyProtocol-policy'

    def _policy_name(self, policy_type):
        return f'ec2-elb-lb-{policy_type}'

    def _get_listener_policies(self):
        """Get a list of listener policies mapped to the LoadBalancerPort"""
        if not self.elb:
            return {}
        listener_descriptions = self.elb.get('ListenerDescriptions', [])
        policies = {l['LoadBalancerPort']: l['PolicyNames'] for l in listener_descriptions}
        return policies

    def _set_listener_policies(self, port, policies):
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.set_load_balancer_policies_of_listener(aws_retry=True, LoadBalancerName=self.name, LoadBalancerPort=port, PolicyNames=list(policies))
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to set load balancer listener policies', port=port, policies=policies)
        return True

    def _get_stickiness_policies(self):
        """Get a list of AppCookieStickinessPolicyType and LBCookieStickinessPolicyType policies"""
        return list((p['PolicyName'] for p in self.elb_policies if p['PolicyTypeName'] in ['AppCookieStickinessPolicyType', 'LBCookieStickinessPolicyType']))

    def _get_app_stickness_policy_map(self):
        """Get a mapping of App Cookie Stickiness policy names to their definitions"""
        policies = self.elb.get('Policies', {}).get('AppCookieStickinessPolicies', [])
        return {p['PolicyName']: p for p in policies}

    def _get_lb_stickness_policy_map(self):
        """Get a mapping of LB Cookie Stickiness policy names to their definitions"""
        policies = self.elb.get('Policies', {}).get('LBCookieStickinessPolicies', [])
        return {p['PolicyName']: p for p in policies}

    def _purge_stickiness_policies(self):
        """Removes all stickiness policies from all Load Balancers"""
        stickiness_policies = set(self._get_stickiness_policies())
        listeners = self.elb['ListenerDescriptions']
        changed = False
        for listener in listeners:
            port = listener['Listener']['LoadBalancerPort']
            policies = set(listener['PolicyNames'])
            new_policies = set(policies - stickiness_policies)
            if policies != new_policies:
                changed |= self._set_listener_policies(port, new_policies)
        return changed

    def _set_stickiness_policies(self):
        if self.stickiness is None:
            return False
        self._update_descriptions()
        if not self.stickiness['enabled']:
            return self._purge_stickiness_policies()
        if self.stickiness['type'] == 'loadbalancer':
            policy_name = self._policy_name('LBCookieStickinessPolicyType')
            expiration = self.stickiness.get('expiration')
            if not expiration:
                expiration = 0
            policy_description = dict(PolicyName=policy_name, CookieExpirationPeriod=expiration)
            existing_policies = self._get_lb_stickness_policy_map()
            add_method = self.client.create_lb_cookie_stickiness_policy
        elif self.stickiness['type'] == 'application':
            policy_name = self._policy_name('AppCookieStickinessPolicyType')
            policy_description = dict(PolicyName=policy_name, CookieName=self.stickiness.get('cookie', 0))
            existing_policies = self._get_app_stickness_policy_map()
            add_method = self.client.create_app_cookie_stickiness_policy
        else:
            self.module.fail_json(msg=f'Unknown stickiness policy {self.stickiness['type']}')
        changed = False
        if policy_name in existing_policies:
            if existing_policies[policy_name] != policy_description:
                changed |= self._purge_stickiness_policies()
        if changed:
            self._update_descriptions()
        changed |= self._set_stickiness_policy(method=add_method, description=policy_description, existing_policies=existing_policies)
        listeners = self.elb['ListenerDescriptions']
        for listener in listeners:
            changed |= self._set_lb_stickiness_policy(listener=listener, policy=policy_name)
        return changed

    def _delete_loadbalancer_policy(self, policy_name):
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.delete_load_balancer_policy(LoadBalancerName=self.name, PolicyName=policy_name)
        except is_boto3_error_code('InvalidConfigurationRequest'):
            return False
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Failed to load balancer policy {policy_name}')
        return True

    def _set_stickiness_policy(self, method, description, existing_policies=None):
        changed = False
        if existing_policies:
            policy_name = description['PolicyName']
            if policy_name in existing_policies:
                if existing_policies[policy_name] == description:
                    return False
                if existing_policies[policy_name] != description:
                    changed |= self._delete_loadbalancer_policy(policy_name)
        self.changed = True
        changed = True
        if self.check_mode:
            return changed
        if not description.get('CookieExpirationPeriod', None):
            description.pop('CookieExpirationPeriod', None)
        try:
            method(aws_retry=True, LoadBalancerName=self.name, **description)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to create load balancer stickiness policy', description=description)
        return changed

    def _set_lb_stickiness_policy(self, listener, policy):
        port = listener['Listener']['LoadBalancerPort']
        stickiness_policies = set(self._get_stickiness_policies())
        changed = False
        policies = set(listener['PolicyNames'])
        new_policies = list(policies - stickiness_policies)
        new_policies.append(policy)
        if policies != set(new_policies):
            changed |= self._set_listener_policies(port, new_policies)
        return changed

    def _get_backend_policies(self):
        """Get a list of backend policies mapped to the InstancePort"""
        if not self.elb:
            return {}
        server_descriptions = self.elb.get('BackendServerDescriptions', [])
        policies = {b['InstancePort']: b['PolicyNames'] for b in server_descriptions}
        return policies

    def _get_proxy_protocol_policy(self):
        """Returns the name of the name of the ProxyPolicy if created"""
        all_proxy_policies = self._get_proxy_policies()
        if not all_proxy_policies:
            return None
        if len(all_proxy_policies) == 1:
            return all_proxy_policies[0]
        return all_proxy_policies

    def _get_proxy_policies(self):
        """Get a list of ProxyProtocolPolicyType policies"""
        return list((p['PolicyName'] for p in self.elb_policies if p['PolicyTypeName'] == 'ProxyProtocolPolicyType'))

    def _get_policy_map(self):
        """Get a mapping of Policy names to their definitions"""
        return {p['PolicyName']: p for p in self.elb_policies}

    def _set_backend_policies(self):
        """Sets policies for all backends"""
        if not self.listeners:
            return False
        backend_policies = self._get_backend_policies()
        proxy_policies = set(self._get_proxy_policies())
        proxy_ports = dict()
        for listener in self.listeners:
            proxy_protocol = listener.get('proxy_protocol', None)
            if proxy_protocol is None:
                next
            instance_port = listener.get('instance_port')
            if proxy_ports.get(instance_port, None) is not None:
                if proxy_ports[instance_port] != proxy_protocol:
                    self.module.fail_json_aws(f'proxy_protocol set to conflicting values for listeners on port {instance_port}')
            proxy_ports[instance_port] = proxy_protocol
        if not proxy_ports:
            return False
        changed = False
        proxy_policy_name = self._proxy_policy_name()
        if any(proxy_ports.values()):
            changed |= self._set_proxy_protocol_policy(proxy_policy_name)
        for port in proxy_ports:
            current_policies = set(backend_policies.get(port, []))
            new_policies = list(current_policies - proxy_policies)
            if proxy_ports[port]:
                new_policies.append(proxy_policy_name)
            changed |= self._set_backend_policy(port, new_policies)
        return changed

    def _set_backend_policy(self, port, policies):
        backend_policies = self._get_backend_policies()
        current_policies = set(backend_policies.get(port, []))
        if current_policies == set(policies):
            return False
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.set_load_balancer_policies_for_backend_server(aws_retry=True, LoadBalancerName=self.name, InstancePort=port, PolicyNames=policies)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to set load balancer backend policies', port=port, policies=policies)
        return True

    def _set_proxy_protocol_policy(self, policy_name):
        """Install a proxy protocol policy if needed"""
        policy_map = self._get_policy_map()
        policy_attributes = [dict(AttributeName='ProxyProtocol', AttributeValue='true')]
        proxy_policy = dict(PolicyName=policy_name, PolicyTypeName='ProxyProtocolPolicyType', PolicyAttributeDescriptions=policy_attributes)
        existing_policy = policy_map.get(policy_name)
        if proxy_policy == existing_policy:
            return False
        if existing_policy is not None:
            self.module.fail_json(msg=f"Unable to configure ProxyProtocol policy. Policy with name {policy_name} already exists and doesn't match.", policy=proxy_policy, existing_policy=existing_policy)
        proxy_policy['PolicyAttributes'] = proxy_policy.pop('PolicyAttributeDescriptions')
        proxy_policy['LoadBalancerName'] = self.name
        self.changed = True
        if self.check_mode:
            return True
        try:
            self.client.create_load_balancer_policy(aws_retry=True, **proxy_policy)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to create load balancer policy', policy=proxy_policy)
        return True

    def _get_instance_ids(self):
        """Get the current list of instance ids installed in the elb"""
        elb = self.elb or {}
        return list((i['InstanceId'] for i in elb.get('Instances', [])))

    def _change_instances(self, method, instances):
        if not instances:
            return False
        self.changed = True
        if self.check_mode:
            return True
        instance_id_list = list(({'InstanceId': i} for i in instances))
        try:
            method(aws_retry=True, LoadBalancerName=self.name, Instances=instance_id_list)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to change instance registration', instances=instance_id_list, name=self.name)
        return True

    def _set_instance_ids(self):
        """Register or deregister instances from an lb instance"""
        new_instances = self.instance_ids or []
        existing_instances = self._get_instance_ids()
        instances_to_add = set(new_instances) - set(existing_instances)
        if self.purge_instance_ids:
            instances_to_remove = set(existing_instances) - set(new_instances)
        else:
            instances_to_remove = []
        changed = False
        changed |= self._change_instances(self.client.register_instances_with_load_balancer, instances_to_add)
        if self.wait:
            self._wait_for_instance_state('instance_in_service', list(instances_to_add))
        changed |= self._change_instances(self.client.deregister_instances_from_load_balancer, instances_to_remove)
        if self.wait:
            self._wait_for_instance_state('instance_deregistered', list(instances_to_remove))
        return changed

    def _get_tags(self):
        tags = self.client.describe_tags(aws_retry=True, LoadBalancerNames=[self.name])
        if not tags:
            return {}
        try:
            tags = tags['TagDescriptions'][0]['Tags']
        except (KeyError, TypeError):
            return {}
        return boto3_tag_list_to_ansible_dict(tags)

    def _add_tags(self, tags_to_set):
        if not tags_to_set:
            return False
        self.changed = True
        if self.check_mode:
            return True
        tags_to_add = ansible_dict_to_boto3_tag_list(tags_to_set)
        self.client.add_tags(LoadBalancerNames=[self.name], Tags=tags_to_add)
        return True

    def _remove_tags(self, tags_to_unset):
        if not tags_to_unset:
            return False
        self.changed = True
        if self.check_mode:
            return True
        tags_to_remove = [dict(Key=tagkey) for tagkey in tags_to_unset]
        self.client.remove_tags(LoadBalancerNames=[self.name], Tags=tags_to_remove)
        return True

    def _set_tags(self):
        """Add/Delete tags"""
        if self.tags is None:
            return False
        try:
            current_tags = self._get_tags()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to get load balancer tags')
        tags_to_set, tags_to_unset = compare_aws_tags(current_tags, self.tags, self.purge_tags)
        changed = False
        try:
            changed |= self._remove_tags(tags_to_unset)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to remove load balancer tags')
        try:
            changed |= self._add_tags(tags_to_set)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to add load balancer tags')
        return changed

    def _validate_stickiness(self, stickiness):
        problem_found = False
        if not stickiness:
            return problem_found
        if not stickiness['enabled']:
            return problem_found
        if stickiness['type'] == 'application':
            if not stickiness.get('cookie'):
                problem_found = True
                self.module.fail_json(msg='cookie must be specified when stickiness type is "application"', stickiness=stickiness)
            if stickiness.get('expiration'):
                self.warn(msg='expiration is ignored when stickiness type is "application"')
        if stickiness['type'] == 'loadbalancer':
            if stickiness.get('cookie'):
                self.warn(msg='cookie is ignored when stickiness type is "loadbalancer"')
        return problem_found

    def _validate_access_logs(self, access_logs):
        problem_found = False
        if not access_logs:
            return problem_found
        if not access_logs['enabled']:
            return problem_found
        if not access_logs.get('s3_location', None):
            problem_found = True
            self.module.fail_json(msg='s3_location must be provided when access_logs.state is "present"')
        return problem_found

    def _validate_creation_requirements(self):
        if self.elb:
            return False
        problem_found = False
        if not self.subnets and (not self.zones):
            problem_found = True
            self.module.fail_json(msg='One of subnets or zones must be provided when creating an ELB')
        if not self.listeners:
            problem_found = True
            self.module.fail_json(msg='listeners must be provided when creating an ELB')
        return problem_found

    def _validate_listeners(self, listeners):
        if not listeners:
            return False
        return any((self._validate_listener(listener) for listener in listeners))

    def _validate_listener(self, listener):
        problem_found = False
        if not listener:
            return problem_found
        for protocol in ['instance_protocol', 'protocol']:
            value = listener.get(protocol, None)
            problem = self._validate_protocol(value)
            problem_found |= problem
            if problem:
                self.module.fail_json(msg=f'Invalid protocol ({value}) in listener', listener=listener)
        return problem_found

    def _validate_health_check(self, health_check):
        if not health_check:
            return False
        protocol = health_check['ping_protocol']
        if self._validate_protocol(protocol):
            self.module.fail_json(msg=f'Invalid protocol ({protocol}) defined in health check', health_check=health_check)
        if protocol.upper() in ['HTTP', 'HTTPS']:
            if not health_check['ping_path']:
                self.module.fail_json(msg='For HTTP and HTTPS health checks a ping_path must be provided', health_check=health_check)
        return False

    def _validate_protocol(self, protocol):
        if not protocol:
            return False
        return protocol.upper() not in ['HTTP', 'HTTPS', 'TCP', 'SSL']

    @AWSRetry.jittered_backoff()
    def _describe_loadbalancer(self, lb_name):
        paginator = self.client.get_paginator('describe_load_balancers')
        return paginator.paginate(LoadBalancerNames=[lb_name]).build_full_result()['LoadBalancerDescriptions']

    def _get_vpc_from_subnets(self, subnets):
        if not subnets:
            return None
        subnet_details = self._describe_subnets(list(subnets))
        vpc_ids = set((subnet['VpcId'] for subnet in subnet_details))
        if not vpc_ids:
            return None
        if len(vpc_ids) > 1:
            self.module.fail_json('Subnets for an ELB may not span multiple VPCs', subnets=subnet_details, vpc_ids=vpc_ids)
        return vpc_ids.pop()

    @AWSRetry.jittered_backoff()
    def _describe_subnets(self, subnet_ids):
        paginator = self.ec2_client.get_paginator('describe_subnets')
        return paginator.paginate(SubnetIds=subnet_ids).build_full_result()['Subnets']

    @AWSRetry.jittered_backoff()
    def _get_ec2_security_group_ids_from_names(self, **params):
        return get_ec2_security_group_ids_from_names(ec2_connection=self.ec2_client, **params)