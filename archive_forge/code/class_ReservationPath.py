from typing import NamedTuple, Union
from google.api_core.exceptions import InvalidArgument
from google.cloud.pubsublite.types.location import CloudZone, CloudRegion
class ReservationPath(NamedTuple):
    project: Union[int, str]
    location: CloudRegion
    name: str

    def __str__(self):
        return f'projects/{self.project}/locations/{self.location}/reservations/{self.name}'

    def to_location_path(self):
        return LocationPath(self.project, self.location)

    @staticmethod
    def parse(to_parse: str) -> 'ReservationPath':
        splits = to_parse.split('/')
        if len(splits) != 6 or splits[0] != 'projects' or splits[2] != 'locations' or (splits[4] != 'reservations'):
            raise InvalidArgument('Reservation path must be formatted like projects/{project_number}/locations/{location}/reservations/{name} but was instead ' + to_parse)
        return ReservationPath(splits[1], CloudRegion.parse(splits[3]), splits[5])