import re
def _prepare_tag(tag):
    _isinstance, _str = (isinstance, str)
    if tag == '{*}*':

        def select(context, result):
            for elem in result:
                if _isinstance(elem.tag, _str):
                    yield elem
    elif tag == '{}*':

        def select(context, result):
            for elem in result:
                el_tag = elem.tag
                if _isinstance(el_tag, _str) and el_tag[0] != '{':
                    yield elem
    elif tag[:3] == '{*}':
        suffix = tag[2:]
        no_ns = slice(-len(suffix), None)
        tag = tag[3:]

        def select(context, result):
            for elem in result:
                el_tag = elem.tag
                if el_tag == tag or (_isinstance(el_tag, _str) and el_tag[no_ns] == suffix):
                    yield elem
    elif tag[-2:] == '}*':
        ns = tag[:-1]
        ns_only = slice(None, len(ns))

        def select(context, result):
            for elem in result:
                el_tag = elem.tag
                if _isinstance(el_tag, _str) and el_tag[ns_only] == ns:
                    yield elem
    else:
        raise RuntimeError(f'internal parser error, got {tag}')
    return select