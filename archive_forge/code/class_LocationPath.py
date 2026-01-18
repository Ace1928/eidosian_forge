from typing import NamedTuple, Union
from google.api_core.exceptions import InvalidArgument
from google.cloud.pubsublite.types.location import CloudZone, CloudRegion
class LocationPath(NamedTuple):
    project: Union[int, str]
    location: Union[CloudRegion, CloudZone]

    def __str__(self):
        return f'projects/{self.project}/locations/{self.location}'

    @staticmethod
    def parse(to_parse: str) -> 'LocationPath':
        splits = to_parse.split('/')
        if len(splits) != 6 or splits[0] != 'projects' or splits[2] != 'locations':
            raise InvalidArgument('Location path must be formatted like projects/{project_number}/locations/{location} but was instead ' + to_parse)
        return LocationPath(splits[1], _parse_location(splits[3]))