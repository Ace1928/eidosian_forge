import logging
import six
def _abbreviate(uri):
    if uri.startswith('urn:ietf:params') and ':netconf:' in uri:
        splitted = uri.split(':')
        if ':capability:' in uri:
            if uri.startswith('urn:ietf:params:xml:ns:netconf'):
                name, version = (splitted[7], splitted[8])
            else:
                name, version = (splitted[5], splitted[6])
            return [':' + name, ':' + name + ':' + version]
        elif ':base:' in uri:
            if uri.startswith('urn:ietf:params:xml:ns:netconf'):
                return [':base', ':base' + ':' + splitted[7]]
            else:
                return [':base', ':base' + ':' + splitted[5]]
    return []