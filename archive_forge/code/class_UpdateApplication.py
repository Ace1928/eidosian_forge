from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class UpdateApplication:
    """Constants used by the update application command."""
    WAIT_FOR_UPDATE_MESSAGE = 'Updating application'
    ADD_TIMELIMIT_SEC = 60
    UPDATE_MASK_DISPLAY_NAME_FIELD_NAME = 'displayName'
    UPDATE_MASK_DESCRIPTION_FIELD_NAME = 'description'
    UPDATE_MASK_CRITICALITY_FIELD_NAME = 'attributes.criticality'
    UPDATE_MASK_ENVIRONMENT_FIELD_NAME = 'attributes.environment'
    UPDATE_MASK_BUSINESS_OWNERS_FIELD_NAME = 'attributes.businessOwners'
    UPDATE_MASK_DEVELOPER_OWNERS_FIELD_NAME = 'attributes.developerOwners'
    UPDATE_MASK_OPERATOR_OWNERS_FIELD_NAME = 'attributes.operatorOwners'