import argparse as argparse_mod
import collections
import copy
import errno
import json
import os
import re
import sys
import typing as ty
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import loading
import platformdirs
import yaml
from openstack import _log
from openstack.config import _util
from openstack.config import cloud_region
from openstack.config import defaults
from openstack.config import vendors
from openstack import exceptions
from openstack import warnings as os_warnings
class OpenStackConfig:
    _cloud_region_class = cloud_region.CloudRegion
    _defaults_module = defaults

    def __init__(self, config_files=None, vendor_files=None, override_defaults=None, force_ipv4=None, envvar_prefix=None, secure_files=None, pw_func=None, session_constructor=None, app_name=None, app_version=None, load_yaml_config=True, load_envvars=True, statsd_host=None, statsd_port=None, statsd_prefix=None, influxdb_config=None):
        self.log = _log.setup_logging('openstack.config')
        self._session_constructor = session_constructor
        self._app_name = app_name
        self._app_version = app_version
        self._load_envvars = load_envvars
        if load_yaml_config:
            if config_files is not None:
                self._config_files = config_files
            else:
                self._config_files = CONFIG_FILES
            if secure_files is not None:
                self._secure_files = secure_files
            else:
                self._secure_files = SECURE_FILES
            if vendor_files is not None:
                self._vendor_files = vendor_files
            else:
                self._vendor_files = VENDOR_FILES
        else:
            self._config_files = []
            self._secure_files = []
            self._vendor_files = []
        config_file_override = self._get_envvar('OS_CLIENT_CONFIG_FILE')
        if config_file_override:
            self._config_files.insert(0, config_file_override)
        secure_file_override = self._get_envvar('OS_CLIENT_SECURE_FILE')
        if secure_file_override:
            self._secure_files.insert(0, secure_file_override)
        self.defaults = self._defaults_module.get_defaults()
        if override_defaults:
            self.defaults.update(override_defaults)
        self.config_filename, self.cloud_config = self._load_config_file()
        _, secure_config = self._load_secure_file()
        if secure_config:
            self.cloud_config = _util.merge_clouds(self.cloud_config, secure_config)
        if not self.cloud_config:
            self.cloud_config = {'clouds': {}}
        if 'clouds' not in self.cloud_config:
            self.cloud_config['clouds'] = {}
        self.extra_config = copy.deepcopy(self.cloud_config)
        self.extra_config.pop('clouds', None)
        client_config = self.cloud_config.get('client', {})
        if force_ipv4 is not None:
            self.force_ipv4 = force_ipv4
        else:
            prefer_ipv6 = get_boolean(self._get_envvar('OS_PREFER_IPV6', client_config.get('prefer_ipv6', client_config.get('prefer-ipv6', True))))
            force_ipv4 = get_boolean(self._get_envvar('OS_FORCE_IPV4', client_config.get('force_ipv4', client_config.get('broken-ipv6', False))))
            self.force_ipv4 = force_ipv4
            if not prefer_ipv6:
                self.force_ipv4 = True
        self.envvar_key = self._get_envvar('OS_CLOUD_NAME', 'envvars')
        if self.envvar_key in self.cloud_config['clouds']:
            raise exceptions.ConfigException('"{0}" defines a cloud named "{1}", but OS_CLOUD_NAME is also set to "{1}". Please rename either your environment based cloud, or one of your file-based clouds.'.format(self.config_filename, self.envvar_key))
        self.default_cloud = self._get_envvar('OS_CLOUD')
        if load_envvars:
            envvars = self._get_os_environ(envvar_prefix=envvar_prefix)
            if envvars:
                self.cloud_config['clouds'][self.envvar_key] = envvars
                if not self.default_cloud:
                    self.default_cloud = self.envvar_key
        if not self.default_cloud and self.cloud_config['clouds']:
            if len(self.cloud_config['clouds'].keys()) == 1:
                self.default_cloud = next(iter(self.cloud_config['clouds'].keys()))
        if not self.cloud_config['clouds']:
            self.cloud_config = dict(clouds=dict(defaults=dict(self.defaults)))
            self.default_cloud = 'defaults'
        self._cache_auth = False
        self._cache_expiration_time = 0
        self._cache_path = CACHE_PATH
        self._cache_class = 'dogpile.cache.null'
        self._cache_arguments: ty.Dict[str, ty.Any] = {}
        self._cache_expirations: ty.Dict[str, int] = {}
        self._influxdb_config = {}
        if 'cache' in self.cloud_config:
            cache_settings = _util.normalize_keys(self.cloud_config['cache'])
            self._cache_auth = get_boolean(cache_settings.get('auth', self._cache_auth))
            self._cache_expiration_time = cache_settings.get('expiration_time', cache_settings.get('max_age', self._cache_expiration_time))
            if self._cache_expiration_time:
                self._cache_class = 'dogpile.cache.memory'
            self._cache_class = self.cloud_config['cache'].get('class', self._cache_class)
            self._cache_path = os.path.expanduser(cache_settings.get('path', self._cache_path))
            self._cache_arguments = cache_settings.get('arguments', self._cache_arguments)
            self._cache_expirations = cache_settings.get('expiration', self._cache_expirations)
        if load_yaml_config:
            metrics_config = self.cloud_config.get('metrics', {})
            statsd_config = metrics_config.get('statsd', {})
            statsd_host = statsd_host or statsd_config.get('host')
            statsd_port = statsd_port or statsd_config.get('port')
            statsd_prefix = statsd_prefix or statsd_config.get('prefix')
            influxdb_cfg = metrics_config.get('influxdb', {})
            if not influxdb_config:
                influxdb_config = influxdb_cfg
            else:
                influxdb_config.update(influxdb_cfg)
        if influxdb_config:
            config = {}
            if 'use_udp' in influxdb_config:
                use_udp = influxdb_config['use_udp']
                if isinstance(use_udp, str):
                    use_udp = use_udp.lower() in ('true', 'yes', '1')
                elif not isinstance(use_udp, bool):
                    use_udp = False
                    self.log.warning('InfluxDB.use_udp value type is not supported. Use one of [true|false|yes|no|1|0]')
                config['use_udp'] = use_udp
            for key in ['host', 'port', 'username', 'password', 'database', 'measurement', 'timeout']:
                if key in influxdb_config:
                    config[key] = influxdb_config[key]
            self._influxdb_config = config
        if load_envvars:
            statsd_host = statsd_host or os.environ.get('STATSD_HOST')
            statsd_port = statsd_port or os.environ.get('STATSD_PORT')
            statsd_prefix = statsd_prefix or os.environ.get('STATSD_PREFIX')
        self._statsd_host = statsd_host
        self._statsd_port = statsd_port
        self._statsd_prefix = statsd_prefix
        self._argv_timeout = False
        self._pw_callback = pw_func

    def _get_os_environ(self, envvar_prefix=None):
        ret = self._defaults_module.get_defaults()
        if not envvar_prefix:
            envvar_prefix = 'OS_'
        environkeys = [k for k in os.environ.keys() if (k.startswith('OS_') or k.startswith(envvar_prefix)) and (not k.startswith('OS_TEST')) and (not k.startswith('OS_STD')) and (not k.startswith('OS_LOG'))]
        for k in environkeys:
            newkey = k.split('_', 1)[-1].lower()
            ret[newkey] = os.environ[k]
        selectors = set(['OS_CLOUD', 'OS_REGION_NAME', 'OS_CLIENT_CONFIG_FILE', 'OS_CLIENT_SECURE_FILE', 'OS_CLOUD_NAME'])
        if set(environkeys) - selectors:
            return ret
        return None

    def _get_envvar(self, key, default=None):
        if not self._load_envvars:
            return default
        return os.environ.get(key, default)

    def get_extra_config(self, key, defaults=None):
        """Fetch an arbitrary extra chunk of config, laying in defaults.

        :param string key: name of the config section to fetch
        :param dict defaults: (optional) default values to merge under the
                                         found config
        """
        defaults = _util.normalize_keys(defaults or {})
        if not key:
            return defaults
        return _util.merge_clouds(defaults, _util.normalize_keys(self.cloud_config.get(key, {})))

    def _load_config_file(self):
        return self._load_yaml_json_file(self._config_files)

    def _load_secure_file(self):
        return self._load_yaml_json_file(self._secure_files)

    def _load_vendor_file(self):
        return self._load_yaml_json_file(self._vendor_files)

    def _load_yaml_json_file(self, filelist):
        for path in filelist:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        if path.endswith('json'):
                            return (path, json.load(f))
                        else:
                            return (path, yaml.safe_load(f))
                except IOError as e:
                    if e.errno == errno.EACCES:
                        continue
        return (None, {})

    def _expand_region_name(self, region_name):
        return {'name': region_name, 'values': {}}

    def _expand_regions(self, regions):
        ret = []
        for region in regions:
            if isinstance(region, dict):
                if 'name' not in region or not {'name', 'values'} >= set(region):
                    raise exceptions.ConfigException('Invalid region entry at: %s' % region)
                if 'values' not in region:
                    region['values'] = {}
                ret.append(copy.deepcopy(region))
            else:
                ret.append(self._expand_region_name(region))
        return ret

    def _get_regions(self, cloud):
        if cloud not in self.cloud_config['clouds']:
            return [self._expand_region_name('')]
        regions = self._get_known_regions(cloud)
        if not regions:
            regions = [self._expand_region_name('')]
        return regions

    def _get_known_regions(self, cloud):
        config = _util.normalize_keys(self.cloud_config['clouds'][cloud])
        if 'regions' in config:
            return self._expand_regions(config['regions'])
        elif 'region_name' in config:
            if isinstance(config['region_name'], list):
                regions = config['region_name']
            else:
                regions = config['region_name'].split(',')
            if len(regions) > 1:
                warnings.warn(f'Comma separated lists in region_name are deprecated. Please use a yaml list in the regions parameter in {self.config_filename} instead.', os_warnings.OpenStackDeprecationWarning)
            return self._expand_regions(regions)
        else:
            new_cloud: ty.Dict[str, ty.Any] = {}
            our_cloud = self.cloud_config['clouds'].get(cloud, {})
            self._expand_vendor_profile(cloud, new_cloud, our_cloud)
            if 'regions' in new_cloud and new_cloud['regions']:
                return self._expand_regions(new_cloud['regions'])
            elif 'region_name' in new_cloud and new_cloud['region_name']:
                return [self._expand_region_name(new_cloud['region_name'])]

    def _get_region(self, cloud=None, region_name=''):
        if region_name is None:
            region_name = ''
        if not cloud:
            return self._expand_region_name(region_name)
        regions = self._get_known_regions(cloud)
        if not regions:
            return self._expand_region_name(region_name)
        if not region_name:
            return regions[0]
        for region in regions:
            if region['name'] == region_name:
                return region
        raise exceptions.ConfigException('Region {region_name} is not a valid region name for cloud {cloud}. Valid choices are {region_list}. Please note that region names are case sensitive.'.format(region_name=region_name, region_list=','.join([r['name'] for r in regions]), cloud=cloud))

    def get_cloud_names(self):
        return self.cloud_config['clouds'].keys()

    def _get_base_cloud_config(self, name, profile=None):
        cloud = dict()
        if name and name not in self.cloud_config['clouds']:
            raise exceptions.ConfigException('Cloud {name} was not found.'.format(name=name))
        our_cloud = self.cloud_config['clouds'].get(name, dict())
        if profile:
            our_cloud['profile'] = profile
        cloud.update(self.defaults)
        self._expand_vendor_profile(name, cloud, our_cloud)
        if 'auth' not in cloud:
            cloud['auth'] = dict()
        _auth_update(cloud, our_cloud)
        if 'cloud' in cloud:
            del cloud['cloud']
        return cloud

    def _expand_vendor_profile(self, name, cloud, our_cloud):
        profile_name = our_cloud.get('profile', our_cloud.get('cloud', None))
        if not profile_name or profile_name == self.envvar_key:
            return
        if 'cloud' in our_cloud:
            warnings.warn(f"{self.config_filename} uses the keyword 'cloud' to reference a known vendor profile. This has been deprecated in favor of the 'profile' keyword.", os_warnings.OpenStackDeprecationWarning)
        vendor_filename, vendor_file = self._load_vendor_file()
        if vendor_file and 'public-clouds' in vendor_file and (profile_name in vendor_file['public-clouds']):
            _auth_update(cloud, vendor_file['public-clouds'][profile_name])
        else:
            profile_data = vendors.get_profile(profile_name)
            if profile_data:
                nested_profile = profile_data.pop('profile', None)
                if nested_profile:
                    nested_profile_data = vendors.get_profile(nested_profile)
                    if nested_profile_data:
                        profile_data = nested_profile_data
                status = profile_data.pop('status', 'active')
                message = profile_data.pop('message', '')
                if status == 'deprecated':
                    warnings.warn(f'{profile_name} is deprecated: {message}', os_warnings.OpenStackDeprecationWarning)
                elif status == 'shutdown':
                    raise exceptions.ConfigException('{profile_name} references a cloud that no longer exists: {message}'.format(profile_name=profile_name, message=message))
                _auth_update(cloud, profile_data)
            else:
                warnings.warn(f"Couldn't find the vendor profile {profile_name} for the cloud {name}", os_warnings.ConfigurationWarning)

    def _project_scoped(self, cloud):
        return 'project_id' in cloud or 'project_name' in cloud or 'project_id' in cloud['auth'] or ('project_name' in cloud['auth'])

    def _validate_networks(self, networks, key):
        value = None
        for net in networks:
            if value and net[key]:
                raise exceptions.ConfigException('Duplicate network entries for {key}: {net1} and {net2}. Only one network can be flagged with {key}'.format(key=key, net1=value['name'], net2=net['name']))
            if not value and net[key]:
                value = net

    def _fix_backwards_networks(self, cloud):
        networks = []
        for net in cloud.get('networks', []):
            name = net.get('name')
            if not name:
                raise exceptions.ConfigException('Entry in network list is missing required field "name".')
            network = dict(name=name, routes_externally=get_boolean(net.get('routes_externally')), nat_source=get_boolean(net.get('nat_source')), nat_destination=get_boolean(net.get('nat_destination')), default_interface=get_boolean(net.get('default_interface')))
            network['routes_ipv4_externally'] = get_boolean(net.get('routes_ipv4_externally', network['routes_externally']))
            network['routes_ipv6_externally'] = get_boolean(net.get('routes_ipv6_externally', network['routes_externally']))
            networks.append(network)
        for key in ('external_network', 'internal_network'):
            external = key.startswith('external')
            if key in cloud and 'networks' in cloud:
                raise exceptions.ConfigException('Both {key} and networks were specified in the config. Please remove {key} from the config and use the network list to configure network behavior.'.format(key=key))
            if key in cloud:
                warnings.warn(f'{key} is deprecated. Please replace with an entry in a dict inside of the networks list with name: {cloud[key]} and routes_externally: {external}', os_warnings.OpenStackDeprecationWarning)
                networks.append(dict(name=cloud[key], routes_externally=external, nat_destination=not external, default_interface=external))
        self._validate_networks(networks, 'nat_destination')
        self._validate_networks(networks, 'default_interface')
        cloud['networks'] = networks
        return cloud

    def _handle_domain_id(self, cloud):
        mappings = {'domain_id': ('user_domain_id', 'project_domain_id'), 'domain_name': ('user_domain_name', 'project_domain_name')}
        for target_key, possible_values in mappings.items():
            if not self._project_scoped(cloud):
                if target_key in cloud and target_key not in cloud['auth']:
                    cloud['auth'][target_key] = cloud.pop(target_key)
                continue
            for key in possible_values:
                if target_key in cloud['auth'] and key not in cloud['auth']:
                    cloud['auth'][key] = cloud['auth'][target_key]
            cloud.pop(target_key, None)
            cloud['auth'].pop(target_key, None)
        return cloud

    def _fix_backwards_project(self, cloud):
        mappings = {'domain_id': ('domain_id', 'domain-id'), 'domain_name': ('domain_name', 'domain-name'), 'user_domain_id': ('user_domain_id', 'user-domain-id'), 'user_domain_name': ('user_domain_name', 'user-domain-name'), 'project_domain_id': ('project_domain_id', 'project-domain-id'), 'project_domain_name': ('project_domain_name', 'project-domain-name'), 'token': ('auth-token', 'auth_token', 'token')}
        if cloud.get('auth_type', None) == 'v2password':
            mappings['tenant_id'] = ('project_id', 'project-id', 'tenant_id', 'tenant-id')
            mappings['tenant_name'] = ('project_name', 'project-name', 'tenant_name', 'tenant-name')
        else:
            mappings['project_id'] = ('tenant_id', 'tenant-id', 'project_id', 'project-id')
            mappings['project_name'] = ('tenant_name', 'tenant-name', 'project_name', 'project-name')
        for target_key, possible_values in mappings.items():
            target = None
            for key in possible_values:
                if key in cloud['auth']:
                    target = str(cloud['auth'][key])
                    del cloud['auth'][key]
                if key in cloud:
                    target = str(cloud[key])
                    del cloud[key]
            if target:
                cloud['auth'][target_key] = target
        return cloud

    def _fix_backwards_auth_plugin(self, cloud):
        mappings = {'auth_type': ('auth_plugin', 'auth_type')}
        for target_key, possible_values in mappings.items():
            target = None
            for key in possible_values:
                if key in cloud:
                    target = cloud[key]
                    del cloud[key]
            cloud[target_key] = target
        return cloud

    def register_argparse_arguments(self, parser, argv, service_keys=None):
        """Register all of the common argparse options needed.

        Given an argparse parser, register the keystoneauth Session arguments,
        the keystoneauth Auth Plugin Options and os-cloud. Also, peek in the
        argv to see if all of the auth plugin options should be registered
        or merely the ones already configured.

        :param argparse.ArgumentParser: parser to attach argparse options to
        :param argv: the arguments provided to the application
        :param string service_keys: Service or list of services this argparse
                                    should be specialized for, if known.
                                    The first item in the list will be used
                                    as the default value for service_type
                                    (optional)

        :raises exceptions.ConfigException if an invalid auth-type is requested
        """
        if service_keys is None:
            service_keys = []
        _fix_argv(argv)
        local_parser = argparse_mod.ArgumentParser(add_help=False)
        for p in (parser, local_parser):
            p.add_argument('--os-cloud', metavar='<name>', default=self._get_envvar('OS_CLOUD', None), help='Named cloud to connect to')
        local_parser.add_argument('--timeout', metavar='<timeout>')
        local_parser.add_argument('--os-token')
        local_parser.add_argument('--os-auth-token')
        options, _args = local_parser.parse_known_args(argv)
        if options.timeout:
            self._argv_timeout = True
        cloud_region = self.get_one(argparse=options, validate=False)
        default_auth_type = cloud_region.config['auth_type']
        try:
            loading.register_auth_argparse_arguments(parser, argv, default=default_auth_type)
        except Exception:
            options, _args = parser.parse_known_args(argv)
            plugin_names = loading.get_available_plugin_names()
            raise exceptions.ConfigException('An invalid auth-type was specified: {auth_type}. Valid choices are: {plugin_names}.'.format(auth_type=options.os_auth_type, plugin_names=','.join(plugin_names)))
        if service_keys:
            primary_service = service_keys[0]
        else:
            primary_service = None
        loading.register_session_argparse_arguments(parser)
        adapter.register_adapter_argparse_arguments(parser, service_type=primary_service)
        for service_key in service_keys:
            parser.add_argument('--{service_key}-api-version'.format(service_key=service_key.replace('_', '-')), help=argparse_mod.SUPPRESS)
            adapter.register_service_adapter_argparse_arguments(parser, service_type=service_key)
        parser.add_argument('--http-timeout', help=argparse_mod.SUPPRESS)
        parser.add_argument('--os-endpoint-type', help=argparse_mod.SUPPRESS)
        parser.add_argument('--endpoint-type', help=argparse_mod.SUPPRESS)

    def _fix_backwards_interface(self, cloud):
        new_cloud = {}
        for key in cloud.keys():
            if key.endswith('endpoint_type'):
                target_key = key.replace('endpoint_type', 'interface')
            else:
                target_key = key
            new_cloud[target_key] = cloud[key]
        return new_cloud

    def _fix_backwards_api_timeout(self, cloud):
        new_cloud = {}
        service_timeout = None
        for key in cloud.keys():
            if key.endswith('timeout') and (not (key == 'timeout' or key == 'api_timeout')):
                service_timeout = cloud[key]
            else:
                new_cloud[key] = cloud[key]
        if service_timeout is not None:
            new_cloud['api_timeout'] = service_timeout
        if self._argv_timeout:
            if 'timeout' in new_cloud and new_cloud['timeout']:
                new_cloud['api_timeout'] = new_cloud.pop('timeout')
        return new_cloud

    def get_all(self):
        clouds = []
        for cloud in self.get_cloud_names():
            for region in self._get_regions(cloud):
                if region:
                    clouds.append(self.get_one(cloud, region_name=region['name']))
        return clouds
    get_all_clouds = get_all

    def _fix_args(self, args=None, argparse=None):
        """Massage the passed-in options

        Replace - with _ and strip os_ prefixes.

        Convert an argparse Namespace object to a dict, removing values
        that are either None or ''.
        """
        if not args:
            args = {}
        if argparse:
            o_dict = vars(argparse)
            parsed_args = dict()
            for k in o_dict:
                if o_dict[k] is not None and o_dict[k] != '':
                    parsed_args[k] = o_dict[k]
            args.update(parsed_args)
        os_args = dict()
        new_args = dict()
        for key, val in iter(args.items()):
            if type(args[key]) is dict:
                new_args[key] = self._fix_args(args[key])
                continue
            key = key.replace('-', '_')
            if key.startswith('os_'):
                os_args[key[3:]] = val
            else:
                new_args[key] = val
        new_args.update(os_args)
        return new_args

    def _find_winning_auth_value(self, opt, config):
        opt_name = opt.name.replace('-', '_')
        if opt_name in config:
            return config[opt_name]
        else:
            deprecated = getattr(opt, 'deprecated', getattr(opt, 'deprecated_opts', []))
            for d_opt in deprecated:
                d_opt_name = d_opt.name.replace('-', '_')
                if d_opt_name in config:
                    return config[d_opt_name]

    def auth_config_hook(self, config):
        """Allow examination of config values before loading auth plugin

        OpenStackClient will override this to perform additional checks
        on auth_type.
        """
        return config

    def _get_auth_loader(self, config):
        if config['auth_type'] in (None, 'None', ''):
            config['auth_type'] = 'none'
        elif config['auth_type'] == 'token_endpoint':
            config['auth_type'] = 'admin_token'
        loader = loading.get_plugin_loader(config['auth_type'])
        if config['auth_type'] == 'v3multifactor':
            loader._methods = config.get('auth_methods')
        return loader

    def _validate_auth(self, config, loader):
        plugin_options = loader.get_options()
        for p_opt in plugin_options:
            winning_value = self._find_winning_auth_value(p_opt, config['auth'])
            if not winning_value:
                winning_value = self._find_winning_auth_value(p_opt, config)
            config = self._clean_up_after_ourselves(config, p_opt, winning_value)
            if winning_value:
                if p_opt.dest is None:
                    good_name = p_opt.name.replace('-', '_')
                    config['auth'][good_name] = winning_value
                else:
                    config['auth'][p_opt.dest] = winning_value
            config = self.option_prompt(config, p_opt)
        return config

    def _validate_auth_correctly(self, config, loader):
        plugin_options = loader.get_options()
        for p_opt in plugin_options:
            winning_value = self._find_winning_auth_value(p_opt, config)
            if not winning_value:
                winning_value = self._find_winning_auth_value(p_opt, config['auth'])
            config = self._clean_up_after_ourselves(config, p_opt, winning_value)
            config = self.option_prompt(config, p_opt)
        return config

    def option_prompt(self, config, p_opt):
        """Prompt user for option that requires a value"""
        if getattr(p_opt, 'prompt', None) is not None and p_opt.dest not in config['auth'] and (self._pw_callback is not None):
            config['auth'][p_opt.dest] = self._pw_callback(p_opt.prompt)
        return config

    def _clean_up_after_ourselves(self, config, p_opt, winning_value):
        for opt in [p_opt.name] + [o.name for o in p_opt.deprecated]:
            opt = opt.replace('-', '_')
            config.pop(opt, None)
            config['auth'].pop(opt, None)
        if winning_value:
            if p_opt.dest is None:
                config['auth'][p_opt.name.replace('-', '_')] = winning_value
            else:
                config['auth'][p_opt.dest] = winning_value
        return config

    def magic_fixes(self, config):
        """Perform the set of magic argument fixups"""
        if 'auth' in config and 'token' in config['auth'] or ('auth_token' in config and config['auth_token']) or ('token' in config and config['token']):
            config.setdefault('token', config.pop('auth_token', None))
        if 'auth' in config and 'passcode' in config:
            config['auth']['passcode'] = config.pop('passcode', None)
        config = self._fix_backwards_api_timeout(config)
        if 'endpoint_type' in config:
            config['interface'] = config.pop('endpoint_type')
        config = self._fix_backwards_auth_plugin(config)
        config = self._fix_backwards_project(config)
        config = self._fix_backwards_interface(config)
        config = self._fix_backwards_networks(config)
        config = self._handle_domain_id(config)
        for key in BOOL_KEYS:
            if key in config:
                if type(config[key]) is not bool:
                    config[key] = get_boolean(config[key])
        for key in CSV_KEYS:
            if key in config:
                if isinstance(config[key], str):
                    config[key] = config[key].split(',')
        if 'auth' in config and 'auth_url' in config['auth']:
            config['auth']['auth_url'] = config['auth']['auth_url'].format(**config)
        return config

    def get_one(self, cloud=None, validate=True, argparse=None, **kwargs):
        """Retrieve a single CloudRegion and merge additional options

        :param string cloud:
            The name of the configuration to load from clouds.yaml
        :param boolean validate:
            Validate the config. Setting this to False causes no auth plugin
            to be created. It's really only useful for testing.
        :param Namespace argparse:
            An argparse Namespace object; allows direct passing in of
            argparse options to be added to the cloud config.  Values
            of None and '' will be removed.
        :param region_name: Name of the region of the cloud.
        :param kwargs: Additional configuration options

        :returns: openstack.config.cloud_region.CloudRegion
        :raises: keystoneauth1.exceptions.MissingRequiredOptions
            on missing required auth parameters
        """
        profile = kwargs.pop('profile', None)
        args = self._fix_args(kwargs, argparse=argparse)
        if cloud is None:
            if 'cloud' in args:
                cloud = args['cloud']
            else:
                cloud = self.default_cloud
        config = self._get_base_cloud_config(cloud, profile)
        if 'region_name' not in args:
            args['region_name'] = ''
        region = self._get_region(cloud=cloud, region_name=args['region_name'])
        args['region_name'] = region['name']
        region_args = copy.deepcopy(region['values'])
        config.pop('regions', None)
        for arg_list in (region_args, args):
            for key, val in iter(arg_list.items()):
                if val is not None:
                    if key == 'auth' and config[key] is not None:
                        config[key] = _auth_update(config[key], val)
                    else:
                        config[key] = val
        config = self.magic_fixes(config)
        config = _util.normalize_keys(config)
        config = self.auth_config_hook(config)
        if validate:
            loader = self._get_auth_loader(config)
            config = self._validate_auth(config, loader)
            auth_plugin = loader.load_from_options(**config['auth'])
        else:
            auth_plugin = None
        for key, value in config.items():
            if hasattr(value, 'format') and key not in FORMAT_EXCLUSIONS:
                config[key] = value.format(**config)
        force_ipv4 = config.pop('force_ipv4', self.force_ipv4)
        prefer_ipv6 = config.pop('prefer_ipv6', True)
        if not prefer_ipv6:
            force_ipv4 = True
        metrics_config = config.get('metrics', {})
        statsd_config = metrics_config.get('statsd', {})
        statsd_host = statsd_config.get('host') or self._statsd_host
        statsd_port = statsd_config.get('port') or self._statsd_port
        statsd_prefix = statsd_config.get('prefix') or self._statsd_prefix
        influxdb_config = metrics_config.get('influxdb', {})
        if influxdb_config:
            merged_influxdb = copy.deepcopy(self._influxdb_config)
            merged_influxdb.update(influxdb_config)
            influxdb_config = merged_influxdb
        else:
            influxdb_config = self._influxdb_config
        if cloud is None:
            cloud_name = ''
        else:
            cloud_name = str(cloud)
        return self._cloud_region_class(name=cloud_name, region_name=config['region_name'], config=config, extra_config=self.extra_config, force_ipv4=force_ipv4, auth_plugin=auth_plugin, openstack_config=self, session_constructor=self._session_constructor, app_name=self._app_name, app_version=self._app_version, cache_auth=self._cache_auth, cache_expiration_time=self._cache_expiration_time, cache_expirations=self._cache_expirations, cache_path=self._cache_path, cache_class=self._cache_class, cache_arguments=self._cache_arguments, password_callback=self._pw_callback, statsd_host=statsd_host, statsd_port=statsd_port, statsd_prefix=statsd_prefix, influxdb_config=influxdb_config)
    get_one_cloud = get_one

    def get_one_cloud_osc(self, cloud=None, validate=True, argparse=None, **kwargs):
        """Retrieve a single CloudRegion and merge additional options

        :param string cloud:
            The name of the configuration to load from clouds.yaml
        :param boolean validate:
            Validate the config. Setting this to False causes no auth plugin
            to be created. It's really only useful for testing.
        :param Namespace argparse:
            An argparse Namespace object; allows direct passing in of
            argparse options to be added to the cloud config.  Values
            of None and '' will be removed.
        :param region_name: Name of the region of the cloud.
        :param kwargs: Additional configuration options

        :raises: keystoneauth1.exceptions.MissingRequiredOptions
            on missing required auth parameters
        """
        args = self._fix_args(kwargs, argparse=argparse)
        if cloud is None:
            if 'cloud' in args:
                cloud = args['cloud']
            else:
                cloud = self.default_cloud
        config = self._get_base_cloud_config(cloud)
        if 'region_name' not in args:
            args['region_name'] = ''
        region = self._get_region(cloud=cloud, region_name=args['region_name'])
        args['region_name'] = region['name']
        region_args = copy.deepcopy(region['values'])
        config.pop('regions', None)
        for arg_list in (region_args, args):
            for key, val in iter(arg_list.items()):
                if val is not None:
                    if key == 'auth' and config[key] is not None:
                        config[key] = _auth_update(config[key], val)
                    else:
                        config[key] = val
        config = self.magic_fixes(config)
        config = self.auth_config_hook(config)
        if validate:
            loader = self._get_auth_loader(config)
            config = self._validate_auth_correctly(config, loader)
            auth_plugin = loader.load_from_options(**config['auth'])
        else:
            auth_plugin = None
        for key, value in config.items():
            if hasattr(value, 'format') and key not in FORMAT_EXCLUSIONS:
                config[key] = value.format(**config)
        force_ipv4 = config.pop('force_ipv4', self.force_ipv4)
        prefer_ipv6 = config.pop('prefer_ipv6', True)
        if not prefer_ipv6:
            force_ipv4 = True
        if cloud is None:
            cloud_name = ''
        else:
            cloud_name = str(cloud)
        return self._cloud_region_class(name=cloud_name, region_name=config['region_name'], config=config, extra_config=self.extra_config, force_ipv4=force_ipv4, auth_plugin=auth_plugin, openstack_config=self, cache_auth=self._cache_auth, cache_expiration_time=self._cache_expiration_time, cache_expirations=self._cache_expirations, cache_path=self._cache_path, cache_class=self._cache_class, cache_arguments=self._cache_arguments, password_callback=self._pw_callback)

    @staticmethod
    def set_one_cloud(config_file, cloud, set_config=None):
        """Set a single cloud configuration.

        :param string config_file:
            The path to the config file to edit. If this file does not exist
            it will be created.
        :param string cloud:
            The name of the configuration to save to clouds.yaml
        :param dict set_config: Configuration options to be set
        """
        set_config = set_config or {}
        cur_config = {}
        try:
            with open(config_file) as fh:
                cur_config = yaml.safe_load(fh)
        except IOError as e:
            if e.errno != 2:
                raise
            pass
        clouds_config = cur_config.get('clouds', {})
        cloud_config = _auth_update(clouds_config.get(cloud, {}), set_config)
        clouds_config[cloud] = cloud_config
        cur_config['clouds'] = clouds_config
        with open(config_file, 'w') as fh:
            yaml.safe_dump(cur_config, fh, default_flow_style=False)