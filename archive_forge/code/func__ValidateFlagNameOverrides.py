from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import info_holders
def _ValidateFlagNameOverrides(self, flag_name_overrides):
    if not flag_name_overrides:
        return
    for attribute_name in flag_name_overrides.keys():
        for attribute in self.concept_spec.attributes:
            if attribute.name == attribute_name:
                break
        else:
            raise ValueError('Attempting to override the name for an attribute not present in the concept: [{}]. Available attributes: [{}]'.format(attribute_name, ', '.join([attribute.name for attribute in self.concept_spec.attributes])))