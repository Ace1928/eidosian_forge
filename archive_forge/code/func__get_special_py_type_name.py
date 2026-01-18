from botocore.utils import is_json_value_header
def _get_special_py_type_name(self, shape):
    special_type_names = {'document_type': ':ref:`document<document>`', 'jsonvalue_header': 'JSON serializable', 'streaming_input_shape': 'bytes or seekable file-like object', 'streaming_output_shape': ':class:`.StreamingBody`', 'eventstream_output_shape': ':class:`.EventStream`'}
    return self._get_value_for_special_type(shape, special_type_names)