from typing import NamedTuple, Union
from google.api_core.exceptions import InvalidArgument
from google.cloud.pubsublite.types.location import CloudZone, CloudRegion
def _parse_location(to_parse: str) -> Union[CloudRegion, CloudZone]:
    try:
        return CloudZone.parse(to_parse)
    except InvalidArgument:
        pass
    try:
        return CloudRegion.parse(to_parse)
    except InvalidArgument:
        pass
    raise InvalidArgument('Invalid location name: ' + to_parse)