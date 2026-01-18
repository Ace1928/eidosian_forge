from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
class NetAppONTAPZapi:
    """ calls a ZAPI command """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_zapi_only_spec()
        self.argument_spec.update(dict(zapi=dict(required=True, type='dict'), vserver=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=False)
        parameters = self.module.params
        self.zapi = parameters['zapi']
        self.vserver = parameters['vserver']
        if not HAS_JSON:
            self.module.fail_json(msg='the python json module is required')
        if not netapp_utils.has_netapp_lib():
            self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
        if not HAS_XMLTODICT:
            self.module.fail_json(msg='the python xmltodict module is required')
        if self.vserver is not None:
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.vserver)
        else:
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)

    def jsonify_and_parse_output(self, xml_data):
        """ convert from XML to JSON
            extract status and error fields is present
        """
        try:
            as_str = xml_data.to_string()
        except Exception as exc:
            self.module.fail_json(msg='Error running zapi in to_string: %s' % str(exc))
        try:
            as_dict = xmltodict.parse(as_str, xml_attribs=True)
        except Exception as exc:
            self.module.fail_json(msg='Error running zapi in xmltodict: %s: %s' % (as_str, str(exc)))
        try:
            as_json = json.loads(json.dumps(as_dict))
        except Exception as exc:
            self.module.fail_json(msg='Error running zapi in json load/dump: %s: %s' % (as_dict, str(exc)))
        if 'results' not in as_json:
            self.module.fail_json(msg='Error running zapi, no results field: %s: %s' % (as_str, repr(as_json)))
        errno = None
        reason = None
        response = as_json.pop('results')
        status = response.get('@status', 'no_status_attr')
        if status != 'passed':
            errno = response.get('@errno', None)
            if errno is None:
                errno = response.get('errorno', None)
            if errno is None:
                errno = 'ESTATUSFAILED'
            reason = response.get('@reason', None)
            if reason is None:
                reason = response.get('reason', None)
            if reason is None:
                reason = 'Execution failure with unknown reason.'
        for key in ('@status', '@errno', '@reason', '@xmlns'):
            try:
                del response[key]
            except KeyError:
                pass
        return (response, status, errno, reason)

    def run_zapi(self):
        """ calls the ZAPI """
        zapi_struct = self.zapi
        error = None
        if not isinstance(zapi_struct, dict):
            error = 'A directory entry is expected, eg: system-get-version: '
            zapi = zapi_struct
        else:
            zapi = list(zapi_struct.keys())
            if len(zapi) != 1:
                error = 'A single ZAPI can be called at a time'
            else:
                zapi = zapi[0]
        if error:
            self.module.fail_json(msg='%s, received: %s' % (error, zapi))
        zapi_obj = netapp_utils.zapi.NaElement(zapi)
        attributes = zapi_struct[zapi]
        if attributes is not None and attributes != 'None':
            zapi_obj.translate_struct(attributes)
        try:
            output = self.server.invoke_elem(zapi_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error running zapi %s: %s' % (zapi, to_native(error)), exception=traceback.format_exc())
        return self.jsonify_and_parse_output(output)

    def apply(self):
        """ calls the zapi and returns json output """
        response, status, errno, reason = self.run_zapi()
        if status == 'passed':
            self.module.exit_json(changed=True, response=response)
        msg = 'ZAPI failure: check errno and reason.'
        self.module.fail_json(changed=False, response=response, status=status, errno=errno, reason=reason, msg=msg)