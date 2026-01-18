from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def __fix_remote_object_internal(self, robject, module_schema, log):
    if type(robject) is not dict:
        return True
    need_bypass = False
    keys_to_delete = list()
    for key in robject:
        value = robject[key]
        if key not in module_schema:
            keys_to_delete.append(key)
            continue
        attr_schema = module_schema[key]
        attr_type = attr_schema['type']
        if attr_type in ['str', 'int']:
            if type(value) is list:
                if len(value) == 1:
                    robject[key] = value[0]
                    log.write('\tfix list-to-atomic key:%s\n' % key)
                else:
                    need_bypass = True
            elif type(value) is dict:
                need_bypass = True
            if not value or value == 'null':
                log.write('\tdelete empty key:%s\n' % key)
                keys_to_delete.append(key)
        elif attr_type == 'dict':
            if 'options' in attr_schema and type(value) is dict:
                need_bypass |= self.__fix_remote_object_internal(value, attr_schema['options'], log)
            else:
                need_bypass = True
            if not value or value == 'null':
                log.write('\tdelete empty key:%s\n' % key)
                keys_to_delete.append(key)
        elif attr_type == 'list':
            if 'options' in attr_schema and type(value) is list:
                for sub_value in value:
                    need_bypass |= self.__fix_remote_object_internal(sub_value, attr_schema['options'], log)
            else:
                need_bypass = True
            if type(value) is list and (not len(value)) or value == 'null' or (not value):
                log.write('\tdelete empty key:%s\n' % key)
                keys_to_delete.append(key)
        else:
            continue
    for key in keys_to_delete:
        log.write('\tdelete unrecognized key:%s\n' % key)
        del robject[key]
    return need_bypass