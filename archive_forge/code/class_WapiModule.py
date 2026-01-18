from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
class WapiModule(WapiBase):
    """ Implements WapiBase for executing a NIOS module """

    def __init__(self, module):
        self.module = module
        provider = module.params['provider']
        try:
            super(WapiModule, self).__init__(provider)
        except Exception as exc:
            self.module.fail_json(msg=to_text(exc))

    def handle_exception(self, method_name, exc):
        """ Handles any exceptions raised
        This method will be called if an InfobloxException is raised for
        any call to the instance of Connector and also, in case of generic
        exception. This method will then gracefully fail the module.
        :args exc: instance of InfobloxException
        """
        if 'text' in exc.response:
            self.module.fail_json(msg=exc.response['text'], type=exc.response['Error'].split(':')[0], code=exc.response.get('code'), operation=method_name)
        else:
            self.module.fail_json(msg=to_native(exc))

    def run(self, ib_obj_type, ib_spec):
        """ Runs the module and performans configuration tasks
        :args ib_obj_type: the WAPI object type to operate against
        :args ib_spec: the specification for the WAPI object as a dict
        :returns: a results dict
        """
        update = new_name = None
        state = self.module.params['state']
        if state not in ('present', 'absent'):
            self.module.fail_json(msg='state must be one of `present`, `absent`, got `%s`' % state)
        result = {'changed': False}
        obj_filter = dict([(k, self.module.params[k]) for k, v in iteritems(ib_spec) if v.get('ib_req')])
        ib_obj_ref, update, new_name = self.get_object_ref(self.module, ib_obj_type, obj_filter, ib_spec)
        if ib_obj_type == NIOS_RANGE and len(ib_obj_ref) == 0 and (True for v in ('new_start_addr', 'new_end_addr') if v in ib_spec.keys()):
            if self.module.params.get('new_start_addr'):
                obj_filter['start_addr'] = self.module.params.get('new_start_addr')
            if self.module.params.get('new_end_addr'):
                obj_filter['end_addr'] = self.module.params.get('new_end_addr')
            ib_obj_ref, update, new_name = self.get_object_ref(self.module, ib_obj_type, obj_filter, ib_spec)
        proposed_object = {}
        for key, value in iteritems(ib_spec):
            if self.module.params[key] is not None:
                if 'transform' in value:
                    proposed_object[key] = value['transform'](self.module)
                else:
                    proposed_object[key] = self.module.params[key]
        if not proposed_object.get('configure_for_dns') and proposed_object.get('view') == 'default' and (ib_obj_type == NIOS_HOST_RECORD):
            del proposed_object['view']
        if ib_obj_ref:
            if len(ib_obj_ref) > 1:
                for each in ib_obj_ref:
                    if each.get('ipv4addr') and each.get('ipv4addr') == proposed_object.get('ipv4addr'):
                        current_object = each
                        break
                    elif each.get('ipv4addrs') and each.get('ipv4addrs')[0].get('ipv4addr') == proposed_object.get('ipv4addrs')[0].get('ipv4addr'):
                        current_object = each
                    else:
                        current_object = obj_filter
                        ref = None
            else:
                current_object = ib_obj_ref[0]
            if 'extattrs' in current_object:
                current_object['extattrs'] = flatten_extattrs(current_object['extattrs'])
            if current_object.get('_ref'):
                ref = current_object.pop('_ref')
        else:
            current_object = obj_filter
            ref = None
        if ib_obj_type == NIOS_MEMBER:
            proposed_object = member_normalize(proposed_object)
            if proposed_object.get('create_token') is not True:
                proposed_object.pop('create_token')
        if ib_obj_type == NIOS_IPV4_NETWORK or ib_obj_type == NIOS_IPV6_NETWORK:
            proposed_object = convert_members_to_struct(proposed_object)
        if ib_obj_type == NIOS_RANGE:
            if proposed_object.get('new_start_addr'):
                proposed_object['start_addr'] = proposed_object.get('new_start_addr')
                del proposed_object['new_start_addr']
            if proposed_object.get('new_end_addr'):
                proposed_object['end_addr'] = proposed_object.get('new_end_addr')
                del proposed_object['new_end_addr']
        if ib_obj_type == NIOS_TXT_RECORD:
            text_obj = proposed_object['text']
            if text_obj.startswith('{'):
                try:
                    text_obj = json.loads(text_obj)
                    txt = text_obj['new_text']
                except Exception:
                    result, exc = safe_eval(text_obj, dict(), include_exceptions=True)
                    if exc is not None:
                        raise TypeError('unable to evaluate string as dictionary')
                    txt = result['new_text']
                proposed_object['text'] = txt
        if update and new_name:
            if ib_obj_type == NIOS_MEMBER:
                proposed_object['host_name'] = new_name
            else:
                proposed_object['name'] = new_name
        check_remove = []
        if ib_obj_type == NIOS_HOST_RECORD:
            if 'ipv4addrs' in [current_object and proposed_object]:
                for each in current_object['ipv4addrs']:
                    if each['ipv4addr'] == proposed_object['ipv4addrs'][0]['ipv4addr']:
                        if 'add' in proposed_object['ipv4addrs'][0]:
                            del proposed_object['ipv4addrs'][0]['add']
                            break
                    check_remove += each.values()
                if proposed_object['ipv4addrs'][0]['ipv4addr'] not in check_remove:
                    if 'remove' in proposed_object['ipv4addrs'][0]:
                        del proposed_object['ipv4addrs'][0]['remove']
        proposed_object = self.check_for_new_ipv4addr(proposed_object)
        res = None
        modified = not self.compare_objects(current_object, proposed_object)
        if 'extattrs' in proposed_object:
            proposed_object['extattrs'] = normalize_extattrs(proposed_object['extattrs'])
        proposed_object = self.check_if_nios_next_ip_exists(proposed_object)
        if state == 'present':
            if ref is None:
                if not self.module.check_mode:
                    self.create_object(ib_obj_type, proposed_object)
                result['changed'] = True
            elif ib_obj_type == NIOS_MEMBER and proposed_object.get('create_token') is True:
                proposed_object = None
                result['api_results'] = self.call_func('create_token', ref, proposed_object)
                result['changed'] = True
            elif modified:
                if 'ipv4addrs' in proposed_object:
                    if 'add' not in proposed_object['ipv4addrs'][0] and 'remove' not in proposed_object['ipv4addrs'][0]:
                        self.check_if_recordname_exists(obj_filter, ib_obj_ref, ib_obj_type, current_object, proposed_object)
                if ib_obj_type in (NIOS_HOST_RECORD, NIOS_NETWORK_VIEW, NIOS_DNS_VIEW):
                    run_update = True
                    proposed_object = self.on_update(proposed_object, ib_spec)
                    if 'ipv4addrs' in proposed_object:
                        if ('add' or 'remove') in proposed_object['ipv4addrs'][0]:
                            run_update, proposed_object = self.check_if_add_remove_ip_arg_exists(proposed_object)
                            if run_update:
                                res = self.update_object(ref, proposed_object)
                                result['changed'] = True
                            else:
                                res = ref
                if ib_obj_type in (NIOS_A_RECORD, NIOS_AAAA_RECORD, NIOS_PTR_RECORD, NIOS_SRV_RECORD, NIOS_NAPTR_RECORD):
                    proposed_object = self.on_update(proposed_object, ib_spec)
                    del proposed_object['view']
                    if not self.module.check_mode:
                        res = self.update_object(ref, proposed_object)
                    result['changed'] = True
                if ib_obj_type in NIOS_ZONE:
                    proposed_object = self.on_update(proposed_object, ib_spec)
                    del proposed_object['zone_format']
                    self.update_object(ref, proposed_object)
                    result['changed'] = True
                elif 'network_view' in proposed_object and ib_obj_type not in (NIOS_IPV4_FIXED_ADDRESS, NIOS_IPV6_FIXED_ADDRESS, NIOS_RANGE):
                    proposed_object.pop('network_view')
                    result['changed'] = True
                if not self.module.check_mode and res is None:
                    proposed_object = self.on_update(proposed_object, ib_spec)
                    self.update_object(ref, proposed_object)
                    result['changed'] = True
        elif state == 'absent':
            if ref is not None:
                if 'ipv4addrs' in proposed_object:
                    if 'remove' in proposed_object['ipv4addrs'][0]:
                        self.check_if_add_remove_ip_arg_exists(proposed_object)
                        self.update_object(ref, proposed_object)
                        result['changed'] = True
                elif not self.module.check_mode:
                    self.delete_object(ref)
                    result['changed'] = True
        return result

    def check_if_recordname_exists(self, obj_filter, ib_obj_ref, ib_obj_type, current_object, proposed_object):
        """ Send POST request if host record input name and retrieved ref name is same,
            but input IP and retrieved IP is different"""
        if 'name' in (obj_filter and ib_obj_ref[0]) and ib_obj_type == NIOS_HOST_RECORD:
            obj_host_name = obj_filter['name']
            ref_host_name = ib_obj_ref[0]['name']
            if 'ipv4addrs' in (current_object and proposed_object):
                current_ip_addr = current_object['ipv4addrs'][0]['ipv4addr']
                proposed_ip_addr = proposed_object['ipv4addrs'][0]['ipv4addr']
            elif 'ipv6addrs' in (current_object and proposed_object):
                current_ip_addr = current_object['ipv6addrs'][0]['ipv6addr']
                proposed_ip_addr = proposed_object['ipv6addrs'][0]['ipv6addr']
            if obj_host_name == ref_host_name and current_ip_addr != proposed_ip_addr:
                self.create_object(ib_obj_type, proposed_object)

    def get_network_view(self, proposed_object):
        """ Check for the associated network view with
            the given dns_view"""
        try:
            network_view_ref = self.get_object('view', {'name': proposed_object['view']}, return_fields=['network_view'])
            if network_view_ref:
                network_view = network_view_ref[0].get('network_view')
        except Exception:
            raise Exception('object with dns_view: %s not found' % proposed_object['view'])
        return network_view

    def check_if_nios_next_ip_exists(self, proposed_object):
        """ Check if nios_next_ip argument is passed in ipaddr while creating
            host record, if yes then format proposed object ipv4addrs and pass
            func:nextavailableip and ipaddr range to create hostrecord with next
             available ip in one call to avoid any race condition """
        if 'ipv4addrs' in proposed_object:
            if 'nios_next_ip' in proposed_object['ipv4addrs'][0]['ipv4addr']:
                ip_range = check_type_dict(proposed_object['ipv4addrs'][0]['ipv4addr'])['nios_next_ip']
                proposed_object['ipv4addrs'][0]['ipv4addr'] = NIOS_NEXT_AVAILABLE_IP + ':' + ip_range
        elif 'ipv4addr' in proposed_object:
            if 'nios_next_ip' in proposed_object['ipv4addr']:
                ip_range = check_type_dict(proposed_object['ipv4addr'])['nios_next_ip']
                net_view = self.get_network_view(proposed_object)
                proposed_object['ipv4addr'] = NIOS_NEXT_AVAILABLE_IP + ':' + ip_range + ',' + net_view
        return proposed_object

    def check_for_new_ipv4addr(self, proposed_object):
        """ Checks if new_ipv4addr parameter is passed in the argument
            while updating the record with new ipv4addr with static allocation"""
        if 'ipv4addr' in proposed_object:
            if 'new_ipv4addr' in proposed_object['ipv4addr']:
                new_ipv4 = check_type_dict(proposed_object['ipv4addr'])['new_ipv4addr']
                proposed_object['ipv4addr'] = new_ipv4
        return proposed_object

    def check_if_add_remove_ip_arg_exists(self, proposed_object):
        """
            This function shall check if add/remove param is set to true and
            is passed in the args, then we will update the proposed dictionary
            to add/remove IP to existing host_record, if the user passes false
            param with the argument nothing shall be done.
            :returns: True if param is changed based on add/remove, and also the
            changed proposed_object.
        """
        update = False
        if 'add' in proposed_object['ipv4addrs'][0]:
            if proposed_object['ipv4addrs'][0]['add']:
                proposed_object['ipv4addrs+'] = proposed_object['ipv4addrs']
                del proposed_object['ipv4addrs']
                del proposed_object['ipv4addrs+'][0]['add']
                update = True
            else:
                del proposed_object['ipv4addrs'][0]['add']
        elif 'remove' in proposed_object['ipv4addrs'][0]:
            if proposed_object['ipv4addrs'][0]['remove']:
                proposed_object['ipv4addrs-'] = proposed_object['ipv4addrs']
                del proposed_object['ipv4addrs']
                del proposed_object['ipv4addrs-'][0]['remove']
                update = True
            else:
                del proposed_object['ipv4addrs'][0]['remove']
        return (update, proposed_object)

    def check_next_ip_status(self, obj_filter):
        """ Checks if nios next ip argument exists if True returns true
            else returns false"""
        if 'ipv4addr' in obj_filter:
            if 'nios_next_ip' in obj_filter['ipv4addr']:
                return True
        return False

    def issubset(self, item, objects):
        """ Checks if item is a subset of objects
        :args item: the subset item to validate
        :args objects: superset list of objects to validate against
        :returns: True if item is a subset of one entry in objects otherwise
            this method will return None
        """
        for obj in objects:
            if isinstance(item, dict):
                if all((entry in obj.items() for entry in item.items())):
                    return True
            elif item in obj:
                return True

    def compare_extattrs(self, current_extattrs, proposed_extattrs):
        """Compare current extensible attributes to given extensible
           attribute, if length is not equal returns false , else
           checks the value of keys in proposed extattrs"""
        if len(current_extattrs) != len(proposed_extattrs):
            return False
        else:
            for key, proposed_item in iteritems(proposed_extattrs):
                current_item = current_extattrs.get(key)
                if current_item != proposed_item:
                    return False
            return True

    def compare_objects(self, current_object, proposed_object):
        for key, proposed_item in iteritems(proposed_object):
            current_item = current_object.get(key)
            if current_item is None:
                return False
            elif isinstance(proposed_item, list):
                if key == 'aliases':
                    if set(current_item) != set(proposed_item):
                        return False
                if key == 'members' and len(proposed_item) != len(current_item):
                    return False
                for subitem in proposed_item:
                    if not self.issubset(subitem, current_item):
                        return False
            elif isinstance(proposed_item, dict):
                if key == 'extattrs':
                    current_extattrs = current_object.get(key)
                    proposed_extattrs = proposed_object.get(key)
                    if not self.compare_extattrs(current_extattrs, proposed_extattrs):
                        return False
                if self.compare_objects(current_item, proposed_item) is False:
                    return False
                else:
                    continue
            elif current_item != proposed_item:
                return False
        return True

    def get_object_ref(self, module, ib_obj_type, obj_filter, ib_spec):
        """ this function gets the reference object of pre-existing nios objects """
        update = False
        old_name = new_name = None
        old_ipv4addr_exists = old_text_exists = False
        next_ip_exists = False
        if 'name' in obj_filter:
            try:
                name_obj = check_type_dict(obj_filter['name'])
                if ib_obj_type == NIOS_NETWORK_VIEW:
                    old_name = name_obj['old_name']
                    new_name = name_obj['new_name']
                else:
                    old_name = name_obj['old_name'].lower()
                    new_name = name_obj['new_name'].lower()
            except TypeError:
                name = obj_filter['name']
            if old_name and new_name:
                if ib_obj_type == NIOS_HOST_RECORD:
                    test_obj_filter = dict([('name', old_name), ('view', obj_filter['view'])])
                elif ib_obj_type == NIOS_A_RECORD:
                    test_obj_filter = dict([('name', old_name), ('ipv4addr', obj_filter['ipv4addr'])])
                    try:
                        ipaddr_obj = check_type_dict(obj_filter['ipv4addr'])
                        ipaddr = ipaddr_obj.get('old_ipv4addr')
                        old_ipv4addr_exists = True if ipaddr else False
                    except TypeError:
                        ipaddr = test_obj_filter['ipv4addr']
                    if old_ipv4addr_exists:
                        test_obj_filter['ipv4addr'] = ipaddr
                    else:
                        del test_obj_filter['ipv4addr']
                else:
                    test_obj_filter = dict([('name', old_name)])
                ib_obj = self.get_object(ib_obj_type, test_obj_filter, return_fields=list(ib_spec.keys()))
                if ib_obj:
                    obj_filter['name'] = new_name
                elif old_ipv4addr_exists and len(ib_obj) == 0:
                    raise Exception("object with name: '%s', ipv4addr: '%s' is not found" % (old_name, test_obj_filter['ipv4addr']))
                else:
                    raise Exception("object with name: '%s' is not found" % old_name)
                update = True
                return (ib_obj, update, new_name)
            if ib_obj_type == NIOS_HOST_RECORD:
                name = obj_filter['name']
                if not obj_filter['configure_for_dns']:
                    test_obj_filter = dict([('name', name)])
                else:
                    test_obj_filter = dict([('name', name), ('view', obj_filter['view'])])
            elif ib_obj_type == NIOS_IPV4_FIXED_ADDRESS and 'mac' in obj_filter:
                test_obj_filter = dict([['mac', obj_filter['mac']]])
            elif ib_obj_type == NIOS_IPV6_FIXED_ADDRESS and 'duid' in obj_filter:
                test_obj_filter = dict([['duid', obj_filter['duid']]])
            elif ib_obj_type == NIOS_CNAME_RECORD:
                test_obj_filter = dict([('name', obj_filter['name']), ('view', obj_filter['view'])])
            elif ib_obj_type == NIOS_A_RECORD:
                test_obj_filter = obj_filter
                test_obj_filter['name'] = test_obj_filter['name'].lower()
                try:
                    ipaddr_obj = check_type_dict(obj_filter['ipv4addr'])
                    ipaddr = ipaddr_obj.get('old_ipv4addr')
                    old_ipv4addr_exists = True if ipaddr else False
                    if not old_ipv4addr_exists:
                        next_ip_exists = self.check_next_ip_status(test_obj_filter)
                except TypeError:
                    ipaddr = obj_filter['ipv4addr']
                if old_ipv4addr_exists:
                    test_obj_filter['ipv4addr'] = ipaddr
                if next_ip_exists:
                    del test_obj_filter['ipv4addr']
            elif ib_obj_type == NIOS_TXT_RECORD:
                test_obj_filter = obj_filter
                try:
                    text_obj = obj_filter['text']
                    if text_obj.startswith('{'):
                        try:
                            text_obj = json.loads(text_obj)
                            txt = text_obj['old_text']
                            old_text_exists = True
                        except Exception:
                            result, exc = safe_eval(text_obj, dict(), include_exceptions=True)
                            if exc is not None:
                                raise TypeError('unable to evaluate string as dictionary')
                            txt = result['old_text']
                            old_text_exists = True
                    else:
                        txt = text_obj
                except TypeError:
                    txt = obj_filter['text']
                test_obj_filter['text'] = txt
            elif ib_obj_type == NIOS_DTC_MONITOR_TCP:
                test_obj_filter = dict([('name', obj_filter['name'])])
            else:
                test_obj_filter = obj_filter
            ib_obj = self.get_object(ib_obj_type, test_obj_filter.copy(), return_fields=list(ib_spec.keys()))
            if old_ipv4addr_exists and (ib_obj is None or len(ib_obj) == 0):
                raise Exception("A Record with ipv4addr: '%s' is not found" % ipaddr)
            if old_text_exists and ib_obj is None:
                raise Exception("TXT Record with text: '%s' is not found" % txt)
        elif ib_obj_type == NIOS_A_RECORD:
            test_obj_filter = obj_filter
            try:
                ipaddr_obj = check_type_dict(obj_filter['ipv4addr'])
                ipaddr = ipaddr_obj.get('old_ipv4addr')
                old_ipv4addr_exists = True if ipaddr else False
            except TypeError:
                ipaddr = obj_filter['ipv4addr']
            test_obj_filter['ipv4addr'] = ipaddr
            ib_obj = self.get_object(ib_obj_type, test_obj_filter.copy(), return_fields=list(ib_spec.keys()))
            if old_ipv4addr_exists and ib_obj is None:
                raise Exception("A Record with ipv4addr: '%s' is not found" % ipaddr)
        elif ib_obj_type == NIOS_TXT_RECORD:
            test_obj_filter = obj_filter
            try:
                text_obj = obj_filter(['text'])
                if text_obj.startswith('{'):
                    try:
                        text_obj = json.loads(text_obj)
                        txt = text_obj['old_text']
                        old_text_exists = True
                    except Exception:
                        result, exc = safe_eval(text_obj, dict(), include_exceptions=True)
                        if exc is not None:
                            raise TypeError('unable to evaluate string as dictionary')
                        txt = result['old_text']
                        old_text_exists = True
                else:
                    txt = text_obj
            except TypeError:
                txt = obj_filter['text']
            test_obj_filter['text'] = txt
            ib_obj = self.get_object(ib_obj_type, test_obj_filter.copy(), return_fields=list(ib_spec.keys()))
            if old_text_exists and ib_obj is None:
                raise Exception("TXT Record with text: '%s' is not found" % txt)
        elif ib_obj_type == NIOS_ZONE:
            temp = ib_spec['restart_if_needed']
            del ib_spec['restart_if_needed']
            ib_obj = self.get_object(ib_obj_type, obj_filter.copy(), return_fields=list(ib_spec.keys()))
            if not ib_obj:
                ib_spec['restart_if_needed'] = temp
        elif ib_obj_type == NIOS_MEMBER:
            test_obj_filter = obj_filter
            try:
                name_obj = check_type_dict(test_obj_filter['host_name'])
                old_name = name_obj['old_name']
                new_name = name_obj['new_name']
            except TypeError:
                host_name = obj_filter['host_name']
            if old_name and new_name:
                test_obj_filter['host_name'] = old_name
                temp = ib_spec['create_token']
                del ib_spec['create_token']
                ib_obj = self.get_object(ib_obj_type, test_obj_filter.copy(), return_fields=list(ib_spec.keys()))
                if temp:
                    ib_spec['create_token'] = temp
                if ib_obj:
                    obj_filter['host_name'] = new_name
                else:
                    raise Exception("object with name: '%s' is not found" % old_name)
                update = True
            else:
                temp = ib_spec['create_token']
                del ib_spec['create_token']
                ib_obj = self.get_object(ib_obj_type, obj_filter.copy(), return_fields=list(ib_spec.keys()))
                if temp:
                    ib_spec['create_token'] = temp
        elif ib_obj_type in (NIOS_IPV4_NETWORK, NIOS_IPV6_NETWORK, NIOS_IPV4_NETWORK_CONTAINER, NIOS_IPV6_NETWORK_CONTAINER):
            temp = ib_spec['template']
            del ib_spec['template']
            if ib_obj_type in (NIOS_IPV4_NETWORK_CONTAINER, NIOS_IPV6_NETWORK_CONTAINER):
                del ib_spec['members']
            ib_obj = self.get_object(ib_obj_type, obj_filter.copy(), return_fields=list(ib_spec.keys()))
            if temp:
                ib_spec['template'] = temp
        elif ib_obj_type in NIOS_RANGE:
            new_start = ib_spec.get('new_start_addr')
            new_end = ib_spec.get('new_end_addr')
            del ib_spec['new_start_addr']
            del ib_spec['new_end_addr']
            new_start_arg = self.module.params.get('new_start_addr')
            new_end_arg = self.module.params.get('new_end_addr')
            ib_obj = self.get_object(ib_obj_type, obj_filter.copy(), return_fields=list(ib_spec.keys()))
            if new_start:
                ib_spec['new_start_addr'] = new_start
            if new_end:
                ib_spec['new_end_addr'] = new_end
            if new_start_arg and new_end_arg:
                if not ib_obj:
                    raise Exception('Specified range %s-%s not found' % (obj_filter['start_addr'], obj_filter['end_addr']))
        else:
            ib_obj = self.get_object(ib_obj_type, obj_filter.copy(), return_fields=list(ib_spec.keys()))
        return (ib_obj, update, new_name)

    def on_update(self, proposed_object, ib_spec):
        """ Event called before the update is sent to the API endpoing
        This method will allow the final proposed object to be changed
        and/or keys filtered before it is sent to the API endpoint to
        be processed.
        :args proposed_object: A dict item that will be encoded and sent
            the API endpoint with the updated data structure
        :returns: updated object to be sent to API endpoint
        """
        keys = set()
        for key, value in iteritems(proposed_object):
            update = ib_spec[key].get('update', True)
            if not update:
                keys.add(key)
        return dict([(k, v) for k, v in iteritems(proposed_object) if k not in keys])