from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
class ROS_api_module:

    def __init__(self):
        module_args = dict(path=dict(type='str', required=True), add=dict(type='str'), remove=dict(type='str'), update=dict(type='str'), cmd=dict(type='str'), query=dict(type='str'), extended_query=dict(type='dict', options=dict(attributes=dict(type='list', elements='str', required=True), where=dict(type='list', elements='dict', options={'attribute': dict(type='str'), 'is': dict(type='str', choices=['==', '!=', '>', '<', 'in', 'eq', 'not', 'more', 'less']), 'value': dict(type='raw'), 'or': dict(type='list', elements='dict', options={'attribute': dict(type='str', required=True), 'is': dict(type='str', choices=['==', '!=', '>', '<', 'in', 'eq', 'not', 'more', 'less'], required=True), 'value': dict(type='raw', required=True)})}, required_together=[('attribute', 'is', 'value')], mutually_exclusive=[('attribute', 'or')], required_one_of=[('attribute', 'or')]))))
        module_args.update(api_argument_spec())
        self.module = AnsibleModule(argument_spec=module_args, supports_check_mode=False, mutually_exclusive=(('add', 'remove', 'update', 'cmd', 'query', 'extended_query'),))
        check_has_library(self.module)
        self.api = create_api(self.module)
        self.path = self.module.params['path'].split()
        self.add = self.module.params['add']
        self.remove = self.module.params['remove']
        self.update = self.module.params['update']
        self.arbitrary = self.module.params['cmd']
        self.where = None
        self.query = self.module.params['query']
        self.extended_query = self.module.params['extended_query']
        self.result = dict(message=[])
        self.api_path = self.api_add_path(self.api, self.path)
        try:
            if self.add:
                self.api_add()
            elif self.remove:
                self.api_remove()
            elif self.update:
                self.api_update()
            elif self.query:
                self.check_query()
                self.api_query()
            elif self.extended_query:
                self.check_extended_query()
                self.api_extended_query()
            elif self.arbitrary:
                self.api_arbitrary()
            else:
                self.api_get_all()
        except UnicodeEncodeError as exc:
            self.module.fail_json(msg='Error while encoding text: {error}'.format(error=exc))

    def check_query(self):
        where_index = self.query.find(' WHERE ')
        if where_index < 0:
            self.query = self.split_params(self.query)
        else:
            where = self.query[where_index + len(' WHERE '):]
            self.query = self.split_params(self.query[:where_index])
            m = re.match('^\\s*([^ ]+)\\s+([^ ]+)\\s+(.*)$', where)
            if not m:
                self.errors("invalid syntax for 'WHERE %s'" % where)
            try:
                self.where = [m.group(1), m.group(2), parse_argument_value(m.group(3).rstrip())[0]]
            except ParseError as exc:
                self.errors("invalid syntax for 'WHERE %s': %s" % (where, exc))
        try:
            idx = self.query.index('WHERE')
            self.where = self.query[idx + 1:]
            self.query = self.query[:idx]
        except ValueError:
            pass

    def check_extended_query_syntax(self, test_atr, or_msg=''):
        if test_atr['is'] == 'in' and (not isinstance(test_atr['value'], list)):
            self.errors("invalid syntax 'extended_query':'where':%s%s 'value' must be a type list" % (or_msg, test_atr))

    def check_extended_query(self):
        if self.extended_query['where']:
            for i in self.extended_query['where']:
                if i['or'] is not None:
                    if len(i['or']) < 2:
                        self.errors("invalid syntax 'extended_query':'where':'or':%s 'or' requires minimum two items" % i['or'])
                    for orv in i['or']:
                        self.check_extended_query_syntax(orv, ":'or':")
                else:
                    self.check_extended_query_syntax(i)

    def list_to_dic(self, ldict):
        return convert_list_to_dictionary(ldict, skip_empty_values=True, require_assignment=True)

    def split_params(self, params):
        if not isinstance(params, str):
            raise AssertionError('Parameters can only be a string, received %s' % type(params))
        try:
            return split_routeros_command(params)
        except ParseError as e:
            self.module.fail_json(msg=to_native(e))

    def api_add_path(self, api, path):
        api_path = api.path()
        for p in path:
            api_path = api_path.join(p)
        return api_path

    def api_get_all(self):
        try:
            for i in self.api_path:
                self.result['message'].append(i)
            self.return_result(False, True)
        except LibRouterosError as e:
            self.errors(e)

    def api_add(self):
        param = self.list_to_dic(self.split_params(self.add))
        try:
            self.result['message'].append('added: .id= %s' % self.api_path.add(**param))
            self.return_result(True)
        except LibRouterosError as e:
            self.errors(e)

    def api_remove(self):
        try:
            self.api_path.remove(self.remove)
            self.result['message'].append('removed: .id= %s' % self.remove)
            self.return_result(True)
        except LibRouterosError as e:
            self.errors(e)

    def api_update(self):
        param = self.list_to_dic(self.split_params(self.update))
        if '.id' not in param.keys():
            self.errors("missing '.id' for %s" % param)
        try:
            self.api_path.update(**param)
            self.result['message'].append('updated: %s' % param)
            self.return_result(True)
        except LibRouterosError as e:
            self.errors(e)

    def api_query(self):
        keys = {}
        for k in self.query:
            if 'id' in k and k != '.id':
                self.errors("'%s' must be '.id'" % k)
            keys[k] = Key(k)
        try:
            if self.where:
                if self.where[1] in ('==', 'eq'):
                    select = self.api_path.select(*keys).where(keys[self.where[0]] == self.where[2])
                elif self.where[1] in ('!=', 'not'):
                    select = self.api_path.select(*keys).where(keys[self.where[0]] != self.where[2])
                elif self.where[1] in ('>', 'more'):
                    select = self.api_path.select(*keys).where(keys[self.where[0]] > self.where[2])
                elif self.where[1] in ('<', 'less'):
                    select = self.api_path.select(*keys).where(keys[self.where[0]] < self.where[2])
                else:
                    self.errors("'%s' is not operator for 'where'" % self.where[1])
            else:
                select = self.api_path.select(*keys)
            for row in select:
                self.result['message'].append(row)
            if len(self.result['message']) < 1:
                msg = "no results for '%s 'query' %s" % (' '.join(self.path), ' '.join(self.query))
                if self.where:
                    msg = msg + ' WHERE %s' % ' '.join(self.where)
                self.result['message'].append(msg)
            self.return_result(False)
        except LibRouterosError as e:
            self.errors(e)

    def build_api_extended_query(self, item):
        if item['attribute'] not in self.extended_query['attributes']:
            self.errors("'%s' attribute is not in attributes: %s" % (item, self.extended_query['attributes']))
        if item['is'] in ('eq', '=='):
            return self.query_keys[item['attribute']] == item['value']
        elif item['is'] in ('not', '!='):
            return self.query_keys[item['attribute']] != item['value']
        elif item['is'] in ('less', '<'):
            return self.query_keys[item['attribute']] < item['value']
        elif item['is'] in ('more', '>'):
            return self.query_keys[item['attribute']] > item['value']
        elif item['is'] == 'in':
            return self.query_keys[item['attribute']].In(*item['value'])
        else:
            self.errors("'%s' is not operator for 'is'" % item['is'])

    def api_extended_query(self):
        self.query_keys = {}
        for k in self.extended_query['attributes']:
            if k == 'id':
                self.errors("'extended_query':'attributes':'%s' must be '.id'" % k)
            self.query_keys[k] = Key(k)
        try:
            if self.extended_query['where']:
                where_args = []
                for i in self.extended_query['where']:
                    if i['or']:
                        where_or_args = []
                        for ior in i['or']:
                            where_or_args.append(self.build_api_extended_query(ior))
                        where_args.append(Or(*where_or_args))
                    else:
                        where_args.append(self.build_api_extended_query(i))
                select = self.api_path.select(*self.query_keys).where(*where_args)
            else:
                select = self.api_path.select(*self.extended_query['attributes'])
            for row in select:
                self.result['message'].append(row)
            self.return_result(False)
        except LibRouterosError as e:
            self.errors(e)

    def api_arbitrary(self):
        param = {}
        self.arbitrary = self.split_params(self.arbitrary)
        arb_cmd = self.arbitrary[0]
        if len(self.arbitrary) > 1:
            param = self.list_to_dic(self.arbitrary[1:])
        try:
            arbitrary_result = self.api_path(arb_cmd, **param)
            for i in arbitrary_result:
                self.result['message'].append(i)
            self.return_result(False)
        except LibRouterosError as e:
            self.errors(e)

    def return_result(self, ch_status=False, status=True):
        if not status:
            self.module.fail_json(msg=self.result['message'])
        else:
            self.module.exit_json(changed=ch_status, msg=self.result['message'])

    def errors(self, e):
        if e.__class__.__name__ == 'TrapError':
            self.result['message'].append('%s' % e)
            self.return_result(False, False)
        self.result['message'].append('%s' % e)
        self.return_result(False, False)