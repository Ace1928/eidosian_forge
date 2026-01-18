from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
class spectrum_model_attrs:

    def __init__(self, module):
        self.module = module
        self.url = module.params['url']
        if not re.search('\\/.+', self.url.split('://')[1]):
            self.url = '%s/spectrum/restful' % self.url.rstrip('/')
        self.attr_map = dict(App_Manufacturer=hex(2295427), CollectionsModelNameString=hex(76507), Condition=hex(65546), Criticality=hex(76044), DeviceType=hex(2293774), isManaged=hex(76125), Model_Class=hex(73448), Model_Handle=hex(76282), Model_Name=hex(65646), Modeltype_Handle=hex(65537), Modeltype_Name=hex(65536), Network_Address=hex(77183), Notes=hex(71012), ServiceDesk_Asset_ID=hex(77241), TopologyModelNameString=hex(76263), sysDescr=hex(65618), sysName=hex(68443), Vendor_Name=hex(71024), Description=hex(2293783))
        self.search_qualifiers = ['and', 'or', 'not', 'greater-than', 'greater-than-or-equals', 'less-than', 'less-than-or-equals', 'equals', 'equals-ignore-case', 'does-not-equal', 'does-not-equal-ignore-case', 'has-prefix', 'does-not-have-prefix', 'has-prefix-ignore-case', 'does-not-have-prefix-ignore-case', 'has-substring', 'does-not-have-substring', 'has-substring-ignore-case', 'does-not-have-substring-ignore-case', 'has-suffix', 'does-not-have-suffix', 'has-suffix-ignore-case', 'does-not-have-suffix-ignore-case', 'has-pcre', 'has-pcre-ignore-case', 'has-wildcard', 'has-wildcard-ignore-case', 'is-derived-from', 'not-is-derived-from']
        self.resp_namespace = dict(ca='http://www.ca.com/spectrum/restful/schema/response')
        self.result = dict(msg='', changed_attrs=dict())
        self.success_msg = 'Success'

    def build_url(self, path):
        """
        Build a sane Spectrum restful API URL
        :param path: The path to append to the restful base
        :type path: str
        :returns: Complete restful API URL
        :rtype: str
        """
        return '%s/%s' % (self.url.rstrip('/'), path.lstrip('/'))

    def attr_id(self, name):
        """
        Get attribute hex ID
        :param name: The name of the attribute to retrieve the hex ID for
        :type name: str
        :returns: Translated hex ID of name, or None if no translation found
        :rtype: str or None
        """
        try:
            return self.attr_map[name]
        except KeyError:
            return None

    def attr_name(self, _id):
        """
        Get attribute name from hex ID
        :param _id: The hex ID to lookup a name for
        :type _id: str
        :returns: Translated name of hex ID, or None if no translation found
        :rtype: str or None
        """
        for name, m_id in list(self.attr_map.items()):
            if _id == m_id:
                return name
        return None

    def urlencode(self, string):
        """
        URL Encode a string
        :param: string: The string to URL encode
        :type string: str
        :returns: URL encode version of supplied string
        :rtype: str
        """
        return quote(string, "<>%-_.!*'():?#/@&+,;=")

    def update_model(self, model_handle, attrs):
        """
        Update a model's attributes
        :param model_handle: The model's handle ID
        :type model_handle: str
        :param attrs: Model's attributes to update. {'<name/id>': '<attr>'}
        :type attrs: dict
        :returns: Nothing; exits on error or updates self.results
        :rtype: None
        """
        update_url = self.build_url('/model/%s?' % model_handle)
        for name, val in list(attrs.items()):
            if val is None:
                val = ''
            val = self.urlencode(str(val))
            if not update_url.endswith('?'):
                update_url += '&'
            update_url += 'attr=%s&val=%s' % (self.attr_id(name) or name, val)
        resp, info = fetch_url(self.module, update_url, method='PUT', headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, use_proxy=self.module.params['use_proxy'])
        status_code = info['status']
        if status_code >= 400:
            body = info['body']
        else:
            body = '' if resp is None else resp.read()
        if status_code != 200:
            self.result['msg'] = 'HTTP PUT error %s: %s: %s' % (status_code, update_url, body)
            self.module.fail_json(**self.result)
        json_resp = json.loads(body)
        '\n        Example success response:\n        {\'model-update-response-list\':{\'model-responses\':{\'model\':{\'@error\':\'Success\',\'@mh\':\'0x1010e76\',\'attribute\':{\'@error\':\'Success\',\'@id\':\'0x1295d\'}}}}}"\n        Example failure response:\n        {\'model-update-response-list\': {\'model-responses\': {\'model\': {\'@error\': \'PartialFailure\', \'@mh\': \'0x1010e76\', \'attribute\': {\'@error-message\': \'brn0vlappua001: You do not have permission to set attribute Network_Address for this model.\', \'@error\': \'Error\', \'@id\': \'0x12d7f\'}}}}}\n        '
        model_resp = json_resp['model-update-response-list']['model-responses']['model']
        if model_resp['@error'] != 'Success':
            self.result['msg'] = str(model_resp['attribute'])
            self.module.fail_json(**self.result)
        self.result['msg'] = self.success_msg
        self.result['changed_attrs'].update(attrs)
        self.result['changed'] = True

    def find_model(self, search_criteria, ret_attrs=None):
        """
        Search for a model in /models
        :param search_criteria: The XML <rs:search-criteria>
        :type search_criteria: str
        :param ret_attrs: List of attributes by name or ID to return back
            (default is Model_Handle)
        :type ret_attrs: list
        returns: Dictionary mapping of ret_attrs to values: {ret_attr: ret_val}
        rtype: dict
        """
        if ret_attrs is None:
            ret_attrs = ['Model_Handle']
        rqstd_attrs = ''
        for ra in ret_attrs:
            _id = self.attr_id(ra) or ra
            rqstd_attrs += '<rs:requested-attribute id="%s" />' % (self.attr_id(ra) or ra)
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n<rs:model-request throttlesize="5"\nxmlns:rs="http://www.ca.com/spectrum/restful/schema/request"\nxmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\nxsi:schemaLocation="http://www.ca.com/spectrum/restful/schema/request ../../../xsd/Request.xsd">\n    <rs:target-models>\n        <rs:models-search>\n            <rs:search-criteria xmlns="http://www.ca.com/spectrum/restful/schema/filter">\n                {0}\n            </rs:search-criteria>\n        </rs:models-search>\n    </rs:target-models>\n {1}\n </rs:model-request>\n'.format(search_criteria, rqstd_attrs)
        url = self.build_url('/models')
        resp, info = fetch_url(self.module, url, data=xml, method='POST', use_proxy=self.module.params['use_proxy'], headers={'Content-Type': 'application/xml', 'Accept': 'application/xml'})
        status_code = info['status']
        if status_code >= 400:
            body = info['body']
        else:
            body = '' if resp is None else resp.read()
        if status_code != 200:
            self.result['msg'] = 'HTTP POST error %s: %s: %s' % (status_code, url, body)
            self.module.fail_json(**self.result)
        root = ET.fromstring(body)
        total_models = int(root.attrib['total-models'])
        error = root.attrib['error']
        model_responses = root.find('ca:model-responses', self.resp_namespace)
        if total_models < 1:
            self.result['msg'] = "No models found matching search criteria `%s'" % search_criteria
            self.module.fail_json(**self.result)
        elif total_models > 1:
            self.result['msg'] = "More than one model found (%s): `%s'" % (total_models, ET.tostring(model_responses, encoding='unicode'))
            self.module.fail_json(**self.result)
        if error != 'EndOfResults':
            self.result['msg'] = "Unexpected search response `%s': %s" % (error, ET.tostring(model_responses, encoding='unicode'))
            self.module.fail_json(**self.result)
        model = model_responses.find('ca:model', self.resp_namespace)
        attrs = model.findall('ca:attribute', self.resp_namespace)
        if not attrs:
            self.result['msg'] = 'No attributes returned.'
            self.module.fail_json(**self.result)
        ret = dict()
        for attr in attrs:
            attr_id = attr.get('id')
            attr_name = self.attr_name(attr_id)
            attr_val = attr.text
            key = attr_name if attr_name in ret_attrs else attr_id
            ret[key] = attr_val
            ret_attrs.remove(key)
        return ret

    def find_model_by_name_type(self, mname, mtype, ret_attrs=None):
        """
        Find a model by name and type
        :param mname: Model name
        :type mname: str
        :param mtype: Model type
        :type mtype: str
        :param ret_attrs: List of attributes by name or ID to return back
            (default is Model_Handle)
        :type ret_attrs: list
        returns: find_model(): Dictionary mapping of ret_attrs to values:
            {ret_attr: ret_val}
        rtype: dict
        """
        if ret_attrs is None:
            ret_attrs = ['Model_Handle']
        'This is basically as follows:\n        <filtered-models>\n            <and>\n                <equals>\n                    <attribute id=...>\n                        <value>...</value>\n                    </attribute>\n                </equals>\n                <equals>\n                    <attribute...>\n                </equals>\n            </and>\n        </filtered-models>\n        '
        filtered_models = ET.Element('filtered-models')
        _and = ET.SubElement(filtered_models, 'and')
        MN_equals = ET.SubElement(_and, 'equals')
        Model_Name = ET.SubElement(MN_equals, 'attribute', {'id': self.attr_map['Model_Name']})
        MN_value = ET.SubElement(Model_Name, 'value')
        MN_value.text = mname
        MTN_equals = ET.SubElement(_and, 'equals')
        Modeltype_Name = ET.SubElement(MTN_equals, 'attribute', {'id': self.attr_map['Modeltype_Name']})
        MTN_value = ET.SubElement(Modeltype_Name, 'value')
        MTN_value.text = mtype
        return self.find_model(ET.tostring(filtered_models, encoding='unicode'), ret_attrs)

    def ensure_model_attrs(self):
        req_attrs = []
        for attr in self.module.params['attributes']:
            req_attrs.append(attr['name'])
        if 'Model_Handle' not in req_attrs:
            req_attrs.append('Model_Handle')
        cur_attrs = self.find_model_by_name_type(self.module.params['name'], self.module.params['type'], req_attrs)
        Model_Handle = cur_attrs.pop('Model_Handle')
        for attr in self.module.params['attributes']:
            req_name = attr['name']
            req_val = attr['value']
            if req_val == '':
                req_val = None
            if cur_attrs[req_name] != req_val:
                if self.module.check_mode:
                    self.result['changed_attrs'][req_name] = req_val
                    self.result['msg'] = self.success_msg
                    self.result['changed'] = True
                    continue
                resp = self.update_model(Model_Handle, {req_name: req_val})
        self.module.exit_json(**self.result)