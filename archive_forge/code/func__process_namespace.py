from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
import platform
from inspect import isgenerator
def _process_namespace(name, namespaces, ns_sep=':', attr_prefix='@'):
    if not namespaces:
        return name
    try:
        ns, name = name.rsplit(ns_sep, 1)
    except ValueError:
        pass
    else:
        ns_res = namespaces.get(ns.strip(attr_prefix))
        name = '{}{}{}{}'.format(attr_prefix if ns.startswith(attr_prefix) else '', ns_res, ns_sep, name) if ns_res else name
    return name