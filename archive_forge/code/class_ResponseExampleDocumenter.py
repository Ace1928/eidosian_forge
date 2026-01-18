from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_default
class ResponseExampleDocumenter(BaseExampleDocumenter):
    EVENT_NAME = 'response-example'

    def document_shape_type_event_stream(self, section, shape, history, **kwargs):
        section.write('EventStream(')
        self.document_shape_type_structure(section, shape, history, **kwargs)
        end_section = section.add_new_section('event-stream-end')
        end_section.write(')')