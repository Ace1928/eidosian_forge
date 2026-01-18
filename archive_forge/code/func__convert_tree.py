import re
from lxml import etree, html
def _convert_tree(beautiful_soup_tree, makeelement):
    if makeelement is None:
        makeelement = html.html_parser.makeelement
    first_element_idx = last_element_idx = None
    html_root = declaration = None
    for i, e in enumerate(beautiful_soup_tree):
        if isinstance(e, Tag):
            if first_element_idx is None:
                first_element_idx = i
            last_element_idx = i
            if html_root is None and e.name and (e.name.lower() == 'html'):
                html_root = e
        elif declaration is None and isinstance(e, _DECLARATION_OR_DOCTYPE):
            declaration = e
    if first_element_idx is None:
        pre_root = post_root = []
        roots = beautiful_soup_tree.contents
    else:
        pre_root = beautiful_soup_tree.contents[:first_element_idx]
        roots = beautiful_soup_tree.contents[first_element_idx:last_element_idx + 1]
        post_root = beautiful_soup_tree.contents[last_element_idx + 1:]
    if html_root is not None:
        i = roots.index(html_root)
        html_root.contents = roots[:i] + html_root.contents + roots[i + 1:]
    else:
        html_root = _PseudoTag(roots)
    convert_node = _init_node_converters(makeelement)
    res_root = convert_node(html_root)
    prev = res_root
    for e in reversed(pre_root):
        converted = convert_node(e)
        if converted is not None:
            prev.addprevious(converted)
            prev = converted
    prev = res_root
    for e in post_root:
        converted = convert_node(e)
        if converted is not None:
            prev.addnext(converted)
            prev = converted
    if declaration is not None:
        try:
            doctype_string = declaration.output_ready()
        except AttributeError:
            doctype_string = declaration.string
        match = _parse_doctype_declaration(doctype_string)
        if not match:
            pass
        else:
            external_id, sys_uri = match.groups()
            docinfo = res_root.getroottree().docinfo
            docinfo.public_id = external_id and external_id[1:-1]
            docinfo.system_url = sys_uri and sys_uri[1:-1]
    return res_root