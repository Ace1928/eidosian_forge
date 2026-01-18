from ._gi import \
def _generate_class_info_doc(info):
    header = '\n:Constructors:\n\n::\n\n'
    doc = ''
    if isinstance(info, StructInfo):
        if info.get_size() > 0:
            doc += '    ' + info.get_name() + '()\n'
    else:
        doc += '    ' + info.get_name() + '(**properties)\n'
    for method_info in info.get_methods():
        if method_info.is_constructor():
            doc += '    ' + _generate_callable_info_doc(method_info) + '\n'
    if doc:
        return header + doc
    else:
        return ''