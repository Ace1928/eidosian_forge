from typing import NamedTuple, Union
from google.api_core.exceptions import InvalidArgument
from google.cloud.pubsublite.types.location import CloudZone, CloudRegion
class TopicPath(NamedTuple):
    project: Union[int, str]
    location: Union[CloudRegion, CloudZone]
    name: str

    def __str__(self):
        return f'projects/{self.project}/locations/{self.location}/topics/{self.name}'

    def to_location_path(self):
        return LocationPath(self.project, self.location)

    @staticmethod
    def parse(to_parse: str) -> 'TopicPath':
        splits = to_parse.split('/')
        if len(splits) != 6 or splits[0] != 'projects' or splits[2] != 'locations' or (splits[4] != 'topics'):
            raise InvalidArgument('Topic path must be formatted like projects/{project_number}/locations/{location}/topics/{name} but was instead ' + to_parse)
        return TopicPath(splits[1], _parse_location(splits[3]), splits[5])