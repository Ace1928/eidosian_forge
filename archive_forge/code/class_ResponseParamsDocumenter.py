from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
class ResponseParamsDocumenter(BaseParamsDocumenter):
    """Generates the description for the response parameters"""
    EVENT_NAME = 'response-params'

    def _add_member_documentation(self, section, shape, name=None, **kwargs):
        name_section = section.add_new_section('param-name')
        name_section.write('- ')
        if name is not None:
            name_section.style.bold('%s' % name)
            name_section.write(' ')
        type_section = section.add_new_section('param-type')
        self._document_non_top_level_param_type(type_section, shape)
        documentation_section = section.add_new_section('param-documentation')
        if shape.documentation:
            documentation_section.style.indent()
            if getattr(shape, 'is_tagged_union', False):
                tagged_union_docs = section.add_new_section('param-tagged-union-docs')
                note = '.. note::    This is a Tagged Union structure. Only one of the     following top level keys will be set: %s.     If a client receives an unknown member it will     set ``SDK_UNKNOWN_MEMBER`` as the top level key,     which maps to the name or tag of the unknown     member. The structure of ``SDK_UNKNOWN_MEMBER`` is     as follows'
                tagged_union_members_str = ', '.join(['``%s``' % key for key in shape.members.keys()])
                unknown_code_example = "'SDK_UNKNOWN_MEMBER': {'name': 'UnknownMemberName'}"
                tagged_union_docs.write(note % tagged_union_members_str)
                example = section.add_new_section('param-unknown-example')
                example.style.codeblock(unknown_code_example)
            documentation_section.include_doc_string(shape.documentation)
        section.style.new_paragraph()

    def document_shape_type_event_stream(self, section, shape, history, **kwargs):
        self.document_shape_type_structure(section, shape, history, **kwargs)