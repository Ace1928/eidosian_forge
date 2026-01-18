from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
@classmethod
def FromRegionResource(cls, region):
    """Create region from a google.cloud.location.Location message."""
    flex = False
    standard = False
    search_api = False
    region_id = region.labels.additionalProperties[0].value
    for p in region.metadata.additionalProperties:
        if p.key == 'flexibleEnvironmentAvailable' and p.value.boolean_value:
            flex = True
        elif p.key == 'standardEnvironmentAvailable' and p.value.boolean_value:
            standard = True
        elif p.key == 'searchApiAvailable' and p.value.boolean_value:
            search_api = True
    return cls(region_id, standard, flex, search_api)