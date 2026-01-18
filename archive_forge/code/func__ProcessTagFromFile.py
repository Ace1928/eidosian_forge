from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def _ProcessTagFromFile(self, tag_template_ref, tag_file):
    """Processes a tag file into the request."""
    try:
        tag = yaml.load_path(tag_file)
        if not isinstance(tag, dict):
            raise InvalidTagFileError('Error parsing tag file: [invalid format]')
    except yaml.YAMLParseError as e:
        raise InvalidTagFileError('Error parsing tag file: [{}]'.format(e))
    tag_template = self.template_service.Get(self.messages.DatacatalogProjectsLocationsTagTemplatesGetRequest(name=tag_template_ref.RelativeName()))
    field_to_field_type = {}
    for additional_property in tag_template.fields.additionalProperties:
        message_type = additional_property.value.type
        field_to_field_type[additional_property.key] = self._GetFieldType(message_type)
    additional_properties = []
    for field_id, field_value in six.iteritems(tag):
        if field_id not in field_to_field_type:
            raise InvalidTagError('Error parsing tag file: [{}] is not a valid field.'.format(field_id))
        additional_properties.append(self.messages.GoogleCloudDatacatalogV1beta1Tag.FieldsValue.AdditionalProperty(key=field_id, value=self._MakeTagField(field_to_field_type[field_id], field_value)))
    return self.messages.GoogleCloudDatacatalogV1beta1Tag.FieldsValue(additionalProperties=additional_properties)