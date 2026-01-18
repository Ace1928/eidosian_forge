from __future__ import absolute_import, division, print_function
import ssl
import atexit
import base64
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils._text import to_text, to_native
class BaseVMwareInventory:

    def __init__(self, hostname, username, password, port, validate_certs, with_tags, display):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port
        self.with_tags = with_tags
        self.validate_certs = validate_certs
        self.content = None
        self.rest_content = None
        self.display = display

    def do_login(self):
        """
        Check requirements and do login
        """
        self.check_requirements()
        self.si, self.content = self._login()
        if self.with_tags:
            self.rest_content = self._login_vapi()

    def _login_vapi(self):
        """
        Login to vCenter API using REST call
        Returns: connection object

        """
        session = requests.Session()
        session.verify = self.validate_certs
        if not self.validate_certs:
            requests.packages.urllib3.disable_warnings()
        server = self.hostname
        if self.port:
            server += ':' + str(self.port)
        client, err = (None, None)
        try:
            client = create_vsphere_client(server=server, username=self.username, password=self.password, session=session)
        except Exception as error:
            err = error
        if client is None:
            msg = 'Failed to login to %s using %s' % (server, self.username)
            if err:
                msg += ' due to : %s' % to_native(err)
            raise AnsibleError(msg)
        return client

    def _login(self):
        """
        Login to vCenter or ESXi server
        Returns: connection object

        """
        if self.validate_certs and (not hasattr(ssl, 'SSLContext')):
            raise AnsibleError('pyVim does not support changing verification mode with python < 2.7.9. Either update python or set validate_certs to false in configuration YAML file.')
        ssl_context = None
        if not self.validate_certs and hasattr(ssl, 'SSLContext'):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            ssl_context.verify_mode = ssl.CERT_NONE
        service_instance = None
        try:
            service_instance = connect.SmartConnect(host=self.hostname, user=self.username, pwd=self.password, sslContext=ssl_context, port=self.port)
        except vim.fault.InvalidLogin as e:
            raise AnsibleParserError('Unable to log on to vCenter or ESXi API at %s:%s as %s: %s' % (self.hostname, self.port, self.username, e.msg))
        except vim.fault.NoPermission as e:
            raise AnsibleParserError('User %s does not have required permission to log on to vCenter or ESXi API at %s:%s : %s' % (self.username, self.hostname, self.port, e.msg))
        except (requests.ConnectionError, ssl.SSLError) as e:
            raise AnsibleParserError('Unable to connect to vCenter or ESXi API at %s on TCP/%s: %s' % (self.hostname, self.port, e))
        except vmodl.fault.InvalidRequest as e:
            raise AnsibleParserError('Failed to get a response from server %s:%s as request is malformed: %s' % (self.hostname, self.port, e.msg))
        except Exception as e:
            raise AnsibleParserError('Unknown error while connecting to vCenter or ESXi API at %s:%s : %s' % (self.hostname, self.port, e))
        if service_instance is None:
            raise AnsibleParserError('Unknown error while connecting to vCenter or ESXi API at %s:%s' % (self.hostname, self.port))
        atexit.register(connect.Disconnect, service_instance)
        return (service_instance, service_instance.RetrieveContent())

    def check_requirements(self):
        """ Check all requirements for this inventory are satisfied"""
        if not HAS_REQUESTS:
            raise AnsibleParserError('Please install "requests" Python module as this is required for VMware Guest dynamic inventory plugin.')
        if not HAS_PYVMOMI:
            raise AnsibleParserError('Please install "PyVmomi" Python module as this is required for VMware Guest dynamic inventory plugin.')
        if HAS_REQUESTS:
            required_version = (2, 3)
            requests_version = requests.__version__.split('.')[:2]
            try:
                requests_major_minor = tuple(map(int, requests_version))
            except ValueError:
                raise AnsibleParserError("Failed to parse 'requests' library version.")
            if requests_major_minor < required_version:
                raise AnsibleParserError("'requests' library version should be >= %s, found: %s." % ('.'.join([str(w) for w in required_version]), requests.__version__))
        if not HAS_VSPHERE and self.with_tags:
            raise AnsibleError("Unable to find 'vSphere Automation SDK' Python library which is required. Please refer this URL for installation steps - https://code.vmware.com/web/sdk/7.0/vsphere-automation-python")
        if not all([self.hostname, self.username, self.password]):
            raise AnsibleError('Missing one of the following : hostname, username, password. Please read the documentation for more information.')

    def get_managed_objects_properties(self, vim_type, properties=None, resources=None, strict=False):
        """
        Look up a Managed Object Reference in vCenter / ESXi Environment
        :param vim_type: Type of vim object e.g, for datacenter - vim.Datacenter
        :param properties: List of properties related to vim object e.g. Name
        :param resources: List of resources to limit search scope
        :param strict: Dictates if plugin raises error or just warns
        :return: local content object
        """
        traversal_spec = vmodl.query.PropertyCollector.TraversalSpec
        filter_spec = vmodl.query.PropertyCollector.FilterSpec
        object_spec = vmodl.query.PropertyCollector.ObjectSpec
        property_spec = vmodl.query.PropertyCollector.PropertySpec
        resource_filters = resources or []
        type_to_name_map = {}

        def _handle_error(message):
            if strict:
                raise AnsibleError(message)
            self.display.warning(message)

        def get_contents(container, vim_types):
            return self.content.propertyCollector.RetrieveContents([filter_spec(objectSet=[object_spec(obj=self.content.viewManager.CreateContainerView(container, vim_types, True), skip=False, selectSet=[traversal_spec(type=vim.view.ContainerView, path='view', skip=False)])], propSet=[property_spec(type=t, all=False, pathSet=['name']) for t in vim_types])])

        def filter_containers(containers, typ, filter_list):
            if len(filter_list) > 0:
                objs = []
                results = []
                found_filters = {}
                for container in containers:
                    results.extend(get_contents(container, [typ]))
                for res in results:
                    if res.propSet[0].val in filter_list:
                        objs.append(res.obj)
                        found_filters[res.propSet[0].val] = True
                for fil in filter_list:
                    if fil not in found_filters:
                        _handle_error('Unable to find %s %s' % (type_to_name_map[typ], fil))
                return objs
            return containers

        def build_containers(containers, vim_type, names, filters):
            filters = filters or []
            if vim_type:
                containers = filter_containers(containers, vim_type, names)
            new_containers = []
            for fil in filters:
                new_filters = None
                for k, v in fil.items():
                    if k == 'resources':
                        new_filters = v
                    else:
                        vim_type = getattr(vim, _snake_to_camel(k, True))
                        names = v
                        type_to_name_map[vim_type] = k.replace('_', ' ')
                new_containers.extend(build_containers(containers, vim_type, names, new_filters))
            if len(filters) > 0:
                return new_containers
            return containers
        containers = build_containers([self.content.rootFolder], None, None, resource_filters)
        if len(containers) == 0:
            return []
        objs_list = [object_spec(obj=self.content.viewManager.CreateContainerView(r, [vim_type], True), selectSet=[traversal_spec(path='view', skip=False, type=vim.view.ContainerView)]) for r in containers]
        is_all = not properties
        property_spec = property_spec(type=vim_type, all=is_all, pathSet=properties)
        filter_spec = filter_spec(objectSet=objs_list, propSet=[property_spec], reportMissingObjectsInResults=False)
        try:
            return self.content.propertyCollector.RetrieveContents([filter_spec])
        except vmodl.query.InvalidProperty as err:
            _handle_error('Invalid property name: %s' % err.name)
        except Exception as err:
            _handle_error("Couldn't retrieve contents from host: %s" % to_native(err))
        return []