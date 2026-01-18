from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class UpdateApplicationWorkload:
    """Provides const values for Update application workload."""
    EMPTY_UPDATE_HELP_TEXT = 'Please specify fields to update.'
    WAIT_FOR_UPDATE_MESSAGE = 'Updating application workload'
    UPDATE_MASK_DISPLAY_NAME_FIELD_NAME = 'displayName'
    UPDATE_MASK_DESCRIPTION_FIELD_NAME = 'description'
    UPDATE_MASK_ATTRIBUTES_FIELD_NAME = 'attributes'
    UPDATE_MASK_ATTR_CRITICALITY_FIELD_NAME = 'attributes.criticality'
    UPDATE_MASK_ATTR_ENVIRONMENT_FIELD_NAME = 'attributes.environment'
    UPDATE_MASK_ATTR_BUSINESS_OWNERS_FIELD_NAME = 'attributes.businessOwners'
    UPDATE_MASK_ATTR_DEVELOPER_OWNERS_FIELD_NAME = 'attributes.developerOwners'
    UPDATE_MASK_ATTR_OPERATOR_OWNERS_FIELD_NAME = 'attributes.operatorOwners'
    UPDATE_TIMELIMIT_SEC = 60