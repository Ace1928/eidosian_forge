from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_default
class BaseExampleDocumenter(ShapeDocumenter):

    def document_example(self, section, shape, prefix=None, include=None, exclude=None):
        """Generates an example based on a shape

        :param section: The section to write the documentation to.

        :param shape: The shape of the operation.

        :param prefix: Anything to be included before the example

        :type include: Dictionary where keys are parameter names and
            values are the shapes of the parameter names.
        :param include: The parameter shapes to include in the documentation.

        :type exclude: List of the names of the parameters to exclude.
        :param exclude: The names of the parameters to exclude from
            documentation.
        """
        history = []
        section.style.new_line()
        section.style.start_codeblock()
        if prefix is not None:
            section.write(prefix)
        self.traverse_and_document_shape(section=section, shape=shape, history=history, include=include, exclude=exclude)
        final_blank_line_section = section.add_new_section('final-blank-line')
        final_blank_line_section.style.new_line()

    def document_recursive_shape(self, section, shape, **kwargs):
        section.write("{'... recursive ...'}")

    def document_shape_default(self, section, shape, history, include=None, exclude=None, **kwargs):
        py_type = self._get_special_py_default(shape)
        if py_type is None:
            py_type = py_default(shape.type_name)
        if self._context.get('streaming_shape') == shape:
            py_type = 'StreamingBody()'
        section.write(py_type)

    def document_shape_type_string(self, section, shape, history, include=None, exclude=None, **kwargs):
        if 'enum' in shape.metadata:
            for i, enum in enumerate(shape.metadata['enum']):
                section.write("'%s'" % enum)
                if i < len(shape.metadata['enum']) - 1:
                    section.write('|')
        else:
            self.document_shape_default(section, shape, history)

    def document_shape_type_list(self, section, shape, history, include=None, exclude=None, **kwargs):
        param_shape = shape.member
        list_section = section.add_new_section('list-value')
        self._start_nested_param(list_section, '[')
        param_section = list_section.add_new_section('member', context={'shape': param_shape.name})
        self.traverse_and_document_shape(section=param_section, shape=param_shape, history=history)
        ending_comma_section = list_section.add_new_section('ending-comma')
        ending_comma_section.write(',')
        ending_bracket_section = list_section.add_new_section('ending-bracket')
        self._end_nested_param(ending_bracket_section, ']')

    def document_shape_type_structure(self, section, shape, history, include=None, exclude=None, **kwargs):
        if not shape.members:
            section.write('{}')
            return
        section = section.add_new_section('structure-value')
        self._start_nested_param(section, '{')
        input_members = self._add_members_to_shape(shape.members, include)
        for i, param in enumerate(input_members):
            if exclude and param in exclude:
                continue
            param_section = section.add_new_section(param)
            param_section.write("'%s': " % param)
            param_shape = input_members[param]
            param_value_section = param_section.add_new_section('member-value', context={'shape': param_shape.name})
            self.traverse_and_document_shape(section=param_value_section, shape=param_shape, history=history, name=param)
            if i < len(input_members) - 1:
                ending_comma_section = param_section.add_new_section('ending-comma')
                ending_comma_section.write(',')
                ending_comma_section.style.new_line()
        self._end_structure(section, '{', '}')

    def document_shape_type_map(self, section, shape, history, include=None, exclude=None, **kwargs):
        map_section = section.add_new_section('map-value')
        self._start_nested_param(map_section, '{')
        value_shape = shape.value
        key_section = map_section.add_new_section('key', context={'shape': shape.key.name})
        key_section.write("'string': ")
        value_section = map_section.add_new_section('value', context={'shape': value_shape.name})
        self.traverse_and_document_shape(section=value_section, shape=value_shape, history=history)
        end_bracket_section = map_section.add_new_section('ending-bracket')
        self._end_nested_param(end_bracket_section, '}')

    def _add_members_to_shape(self, members, include):
        if include:
            members = members.copy()
            for param in include:
                members[param.name] = param
        return members

    def _start_nested_param(self, section, start=None):
        if start is not None:
            section.write(start)
        section.style.indent()
        section.style.indent()
        section.style.new_line()

    def _end_nested_param(self, section, end=None):
        section.style.dedent()
        section.style.dedent()
        section.style.new_line()
        if end is not None:
            section.write(end)

    def _end_structure(self, section, start, end):
        if not section.available_sections:
            section.clear_text()
            section.write(start + end)
            self._end_nested_param(section)
        else:
            end_bracket_section = section.add_new_section('ending-bracket')
            self._end_nested_param(end_bracket_section, end)