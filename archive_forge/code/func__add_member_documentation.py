from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def _add_member_documentation(self, section, shape, name=None, is_top_level_param=False, is_required=False, **kwargs):
    py_type = self._get_special_py_type_name(shape)
    if py_type is None:
        py_type = py_type_name(shape.type_name)
    if is_top_level_param:
        type_section = section.add_new_section('param-type')
        type_section.write(f':type {name}: {py_type}')
        end_type_section = type_section.add_new_section('end-param-type')
        end_type_section.style.new_line()
        name_section = section.add_new_section('param-name')
        name_section.write(':param %s: ' % name)
    else:
        name_section = section.add_new_section('param-name')
        name_section.write('- ')
        if name is not None:
            name_section.style.bold('%s' % name)
            name_section.write(' ')
        type_section = section.add_new_section('param-type')
        self._document_non_top_level_param_type(type_section, shape)
    if is_required:
        is_required_section = section.add_new_section('is-required')
        is_required_section.style.indent()
        is_required_section.style.bold('[REQUIRED]')
        is_required_section.write(' ')
    if shape.documentation:
        documentation_section = section.add_new_section('param-documentation')
        documentation_section.style.indent()
        if getattr(shape, 'is_tagged_union', False):
            tagged_union_docs = section.add_new_section('param-tagged-union-docs')
            note = '.. note::    This is a Tagged Union structure. Only one of the     following top level keys can be set: %s. '
            tagged_union_members_str = ', '.join(['``%s``' % key for key in shape.members.keys()])
            tagged_union_docs.write(note % tagged_union_members_str)
        documentation_section.include_doc_string(shape.documentation)
        self._add_special_trait_documentation(documentation_section, shape)
    end_param_section = section.add_new_section('end-param')
    end_param_section.style.new_paragraph()